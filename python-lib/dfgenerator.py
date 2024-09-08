import json
from json import JSONDecodeError
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class DfGenerator(Dataset):
    """Pandas 데이터프레임과 함께 작동하도록 설계된 YOLO용 커스텀 데이터 생성기입니다."""

    def __init__(self, df_data, class_mapping, cols, base_dir='', transform=None):
        self.base_dir = base_dir  # 이미지 파일 경로의 기본 디렉토리 설정
        self.cols = cols  # 데이터프레임의 열 정보(컬럼명) 설정
        self.classes = class_mapping  # 클래스 매핑 정보를 설정
        self.labels = {v: k for k, v in self.classes.items()}  # 클래스 매핑 정보를 역으로 변환하여 레이블 정보를 생성

        self.image_data = self._read_data(df_data)  # 데이터프레임으로부터 이미지 데이터를 읽어옵니다
        self.image_names = list(self.image_data.keys())  # 이미지 파일 이름 리스트 생성
        self.transform = transform  # 이미지 변환 함수

    def _read_data(self, df):
        """데이터프레임으로부터 이미지 데이터와 레이블을 읽어옵니다."""
        def assert_and_retrieve(obj, prop):
            """레이블 JSON 객체에서 특정 속성을 확인하고 그 값을 반환합니다."""
            if prop not in obj:
                raise Exception(f"Property {prop} not found in label JSON object")
            return obj[prop]

        data = {}
        for _, row in df.iterrows():  # 데이터프레임의 각 행에 대해 반복합니다
            img_file = row[self.cols['col_filename']]  # 이미지 파일 이름 가져오기
            label_data = row[self.cols['col_label']]  # 레이블 데이터 가져오기
            if img_file.startswith('.') or img_file.startswith('/'):  # 파일 이름이 '.' 또는 '/'로 시작하면 제거
                img_file = img_file[1:]

            if img_file not in data:  # 데이터에 이미지 파일이 없으면 빈 리스트 추가
                data[img_file] = []

            if self.cols['single_column_data']:  # 레이블 데이터가 단일 JSON 열에 저장된 경우
                try:
                    label_data_obj = json.loads(label_data)  # 레이블 데이터를 JSON으로 파싱
                except JSONDecodeError as e:
                    raise Exception(f"Failed to parse label JSON: {label_data}") from e

                for label in label_data_obj:  # 각 레이블에 대해 반복
                    y1 = assert_and_retrieve(label, "top")
                    x1 = assert_and_retrieve(label, "left")
                    x2 = x1 + assert_and_retrieve(label, "width")
                    y2 = y1 + assert_and_retrieve(label, "height")
                    data[img_file].append({
                        'x1': int(x1), 'x2': int(x2),
                        'y1': int(y1), 'y2': int(y2),
                        'class': assert_and_retrieve(label, "label")
                    })
            else:  # 레이블 데이터가 개별 열에 저장된 경우
                x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]
                x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]

                if not isinstance(label_data, str) and np.isnan(label_data):
                    continue

                data[img_file].append({
                    'x1': int(x1), 'x2': int(x2),
                    'y1': int(y1), 'y2': int(y2),
                    'class': label_data
                })
        return data

    def __len__(self):
        """전체 이미지의 수를 반환합니다."""
        return len(self.image_names)

    def __getitem__(self, idx):
        """지정된 인덱스에 대한 이미지와 레이블을 반환합니다."""
        img_name = self.image_names[idx]  # 이미지 파일 이름
        img_path = os.path.join(self.base_dir, img_name)  # 이미지 파일 경로
        image = Image.open(img_path).convert('RGB')  # 이미지를 RGB 모드로 열기

        # 이미지와 레이블 데이터
        labels = self.image_data[img_name]

        # 변환이 지정된 경우 적용
        if self.transform:
            image = self.transform(image)

        # YOLO 형식에 맞게 레이블을 변환합니다.
        boxes = []
        class_labels = []
        for label in labels:
            boxes.append([label['x1'], label['y1'], label['x2'], label['y2']])
            class_labels.append(self.classes[label['class']])

        # Tensor로 변환합니다.
        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_labels = torch.tensor(class_labels, dtype=torch.int64)

        # 이미지와 레이블을 튜플로 반환합니다.
        return image, (boxes, class_labels)
