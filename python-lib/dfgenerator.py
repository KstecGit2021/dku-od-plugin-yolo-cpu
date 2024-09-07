import json
from json import JSONDecodeError

import numpy as np
from keras_retinanet.preprocessing.csv_generator import CSVGenerator  # Keras RetinaNet의 CSVGenerator 클래스를 가져옵니다.
from keras_retinanet.preprocessing.generator import Generator  # Keras RetinaNet의 Generator 클래스를 가져옵니다. (경로가 맞는지 확인 필요)

class DfGenerator(CSVGenerator):
    """메모리 내에 있는 Pandas 데이터프레임과 함께 작동하도록 설계된 커스텀 데이터 생성기입니다."""

    def __init__(self, df_data, class_mapping, cols, base_dir='', **kwargs):
        self.base_dir = base_dir  # 이미지 파일 경로의 기본 디렉토리를 설정합니다.
        self.cols = cols  # 데이터프레임의 열 정보(컬럼명)를 설정합니다.
        self.classes = class_mapping  # 클래스 매핑 정보를 설정합니다.
        self.labels = {v: k for k, v in self.classes.items()}  # 클래스 매핑 정보를 역으로 변환하여 레이블 정보를 생성합니다.

        self.image_data = self._read_data(df_data)  # 데이터프레임으로부터 이미지 데이터를 읽어옵니다.
        self.image_names = list(self.image_data.keys())  # 이미지 파일 이름의 리스트를 생성합니다.

        super(DfGenerator, self).__init__(**kwargs)  # 부모 클래스의 초기화 메서드를 호출합니다.

    def _read_classes(self, df):
        """데이터프레임으로부터 클래스 정보를 읽어옵니다."""
        # 데이터프레임의 첫 번째 열을 키로, 두 번째 열을 값으로 하는 딕셔너리를 반환합니다.
        return {row[0]: row[1] for _, row in df.iterrows()}  

    def __len__(self):
        """전체 이미지의 수를 반환합니다."""
        # 이미지 이름 리스트의 길이를 반환하여 전체 이미지 수를 제공합니다.
        return len(self.image_names)  

    def _read_data(self, df):
        """데이터프레임으로부터 이미지 데이터와 레이블을 읽어옵니다."""
        def assert_and_retrieve(obj, prop):
            """레이블 JSON 객체에서 특정 속성을 확인하고 그 값을 반환합니다."""
            if prop not in obj:
                # 속성이 없으면 예외를 발생시킵니다.
                raise Exception(f"Property {prop} not found in label JSON object")  
            return obj[prop]  # 속성 값이 존재하면 그 값을 반환합니다.

        data = {}
        for _, row in df.iterrows():  # 데이터프레임의 각 행에 대해 반복합니다.
            img_file = row[self.cols['col_filename']]  # 이미지 파일 이름을 가져옵니다.
            label_data = row[self.cols['col_label']]  # 레이블 데이터를 가져옵니다.
            if img_file.startswith('.') or img_file.startswith('/'):  # 파일 이름이 '.' 또는 '/'로 시작하면 제거합니다.
                img_file = img_file[1:]

            if img_file not in data:  # 데이터에 이미지 파일이 없으면 빈 리스트를 추가합니다.
                data[img_file] = []

            if self.cols['single_column_data']:  # 레이블 데이터가 단일 JSON 열에 저장된 경우
                try:
                    label_data_obj = json.loads(label_data)  # 레이블 데이터를 JSON으로 파싱합니다.
                except JSONDecodeError as e:
                    # JSON 파싱 실패 시 예외를 발생시킵니다.
                    raise Exception(f"Failed to parse label JSON: {label_data}") from e  

                for label in label_data_obj:  # 각 레이블에 대해 반복합니다.
                    y1 = assert_and_retrieve(label, "top")  # 상단 좌표를 가져옵니다.
                    x1 = assert_and_retrieve(label, "left")  # 좌측 좌표를 가져옵니다.
                    x2 = x1 + assert_and_retrieve(label, "width")  # 우측 좌표를 계산합니다.
                    y2 = y1 + assert_and_retrieve(label, "height")  # 하단 좌표를 계산합니다.
                    data[img_file].append({
                        'x1': int(x1), 'x2': int(x2),
                        'y1': int(y1), 'y2': int(y2),
                        'class': assert_and_retrieve(label, "label")  # 레이블 정보를 추가합니다.
                    })
            else:  # 레이블 데이터가 개별 열에 저장된 경우
                x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]  # 좌측 상단 좌표를 가져옵니다.
                x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]  # 우측 하단 좌표를 가져옵니다.

                # 레이블이 없는 이미지는 건너뜁니다.
                if not isinstance(label_data, str) and np.isnan(label_data):
                    continue

                data[img_file].append({
                    'x1': int(x1), 'x2': int(x2),
                    'y1': int(y1), 'y2': int(y2),
                    'class': label_data  # 레이블 정보를 추가합니다.
                })
        return data  # 수집된 데이터를 반환합니다.
