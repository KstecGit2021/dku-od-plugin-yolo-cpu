import json
from json import JSONDecodeError

import numpy as np
from keras_retinanet.preprocessing.csv_generator import CSVGenerator, Generator


class DfGenerator(CSVGenerator):
    """메모리에 있는 Pandas 데이터프레임과 함께 작동하도록 설계된 커스텀 제너레이터."""

    def __init__(self, df_data, class_mapping, cols, base_dir='', **kwargs):
        self.base_dir = base_dir  # 이미지 파일 경로의 기본 디렉토리 설정
        self.cols = cols  # 데이터프레임의 열 정보 설정
        self.classes = class_mapping  # 클래스 매핑 정보 설정
        self.labels = {v: k for k, v in self.classes.items()}  # 클래스 매핑을 역으로 변환하여 라벨 정보 생성

        self.image_data = self._read_data(df_data)  # 데이터프레임으로부터 이미지 데이터를 읽어옴
        self.image_names = list(self.image_data.keys())  # 이미지 파일 이름 리스트 생성

        Generator.__init__(self, **kwargs)  # 부모 클래스의 초기화 함수 호출

    def _read_classes(self, df):
        """데이터프레임으로부터 클래스를 읽어오는 함수."""
        return {row[0]: row[1] for _, row in df.iterrows()}  # 데이터프레임의 첫 번째 열을 키, 두 번째 열을 값으로 하는 딕셔너리 반환

    def __len__(self):
        """전체 이미지 수를 반환하는 함수."""
        return len(self.image_names)  # 이미지 이름 리스트의 길이를 반환

    def _read_data(self, df):
        """데이터프레임으로부터 이미지 데이터와 라벨을 읽어오는 함수."""
        def assert_and_retrieve(obj, prop):
            """라벨 JSON 객체에서 특정 속성을 확인하고 가져오는 함수."""
            if prop not in obj:
                raise Exception(f"Property {prop} not found in label JSON object")  # 속성이 없으면 예외 발생
            return obj[prop]  # 속성이 있으면 반환

        data = {}
        for _, row in df.iterrows():  # 데이터프레임의 각 행에 대해 반복
            img_file = row[self.cols['col_filename']]  # 이미지 파일 이름 가져오기
            label_data = row[self.cols['col_label']]  # 라벨 데이터 가져오기
            if img_file[0] == '.' or img_file[0] == '/':  # 파일 이름이 '.' 또는 '/'로 시작하면 제거
                img_file = img_file[1:]

            if img_file not in data:  # 데이터에 이미지 파일이 없으면 추가
                data[img_file] = []

            if self.cols['single_column_data']:  # 라벨 데이터가 JSON 형식으로 단일 열에 저장된 경우
                try:
                    label_data_obj = json.loads(label_data)  # 라벨 데이터를 JSON으로 파싱
                except JSONDecodeError as e:
                    raise Exception(f"Failed to parse label JSON: {label_data}") from e  # 파싱 실패 시 예외 발생

                for label in label_data_obj:  # 각 라벨에 대해 반복
                    y1 = assert_and_retrieve(label, "top")  # 상단 좌표 가져오기
                    x1 = assert_and_retrieve(label, "left")  # 좌측 좌표 가져오기
                    x2 = assert_and_retrieve(label, "left") + assert_and_retrieve(label, "width")  # 우측 좌표 계산
                    y2 = assert_and_retrieve(label, "top") + assert_and_retrieve(label, "height")  # 하단 좌표 계산
                    data[img_file].append({
                        'x1': int(x1), 'x2': int(x2),
                        'y1': int(y1), 'y2': int(y2),
                        'class': assert_and_retrieve(label, "label")  # 라벨 추가
                    })
            else:  # 라벨 데이터가 개별 열에 저장된 경우
                x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]  # 좌측 상단 좌표 가져오기
                x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]  # 우측 하단 좌표 가져오기

                # 라벨이 없는 이미지는 건너뜀
                if not isinstance(label_data, str) and np.isnan(label_data): continue

                data[img_file].append({
                    'x1': int(x1), 'x2': int(x2),
                    'y1': int(y1), 'y2': int(y2),
                    'class': label_data  # 라벨 추가
                })
        return data  # 데이터 반환
