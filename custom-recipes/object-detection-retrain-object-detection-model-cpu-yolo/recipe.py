# -*- coding: utf-8 -*-
import constants  # 상수를 정의하는 모듈
import dataiku  # Dataiku 플랫폼 통합 모듈
import gpu_utils  # GPU 유틸리티 함수 모듈
import json  # JSON 처리 모듈
import logging  # 로깅 모듈
import misc_utils  # 다양한 유틸리티 함수 모듈
import numpy as np  # 수치 연산 모듈
import os.path as op  # OS 경로 유틸리티 모듈
import pandas as pd  # 데이터 처리 모듈
import torch  # PyTorch 모듈
import torchvision.transforms as transforms  # 이미지 변환 모듈
import yaml  # YAML 처리 모듈
from dataiku import pandasutils as pdu  # Dataiku pandas 유틸리티 모듈
from dataiku.customrecipe import *  # Dataiku 레시피 API 모듈
from dfgenerator import DfGenerator  # 사용자 정의 DataFrame 생성기
from json import JSONDecodeError  # JSON 디코딩 에러 모듈
from ultralytics import YOLO  # YOLOv5의 최신 업데이트 모듈
import cv2  # OpenCV 모듈
from pathlib import Path  # 파일 경로 처리 모듈

# 로깅을 설정합니다. 로깅 레벨을 INFO로 설정하고, 로그 메시지 형식을 지정합니다.
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

# Dataiku에서 이미지 폴더를 가져옵니다.
images_folder = dataiku.Folder(get_input_names_for_role('images')[0])

# 바운딩 박스 데이터셋을 DataFrame으로 로드합니다.
bb_df = dataiku.Dataset(get_input_names_for_role('bounding_boxes')[0]).get_dataframe()

# Dataiku에서 가중치 폴더를 가져옵니다.
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정합니다. YOLOv5는 .pt 파일을 사용합니다.
weights = op.join(weights_folder.get_path(), 'weights.pt')

# Dataiku에서 출력 폴더를 가져옵니다.
output_folder = dataiku.Folder(get_output_names_for_role('model')[0])

# 모델 가중치의 출력 경로를 설정합니다.
output_path = op.join(output_folder.get_path(), 'weights.pt')

# 레시피 구성을 로드합니다.
configs = get_recipe_config()

# 'should_use_gpu'가 configs에 존재하지 않을 경우 기본값 False를 사용하도록 설정
should_use_gpu = configs.get('should_use_gpu', False)

# GPU 옵션을 로드합니다. GPU 사용 여부, GPU 목록, GPU 할당량을 설정합니다.
gpu_opts = gpu_utils.load_gpu_options(should_use_gpu,
                                      configs.get('list_gpu', ''), 
                                      configs.get('gpu_allocation', 0.))

# 이미지 변환을 정의합니다. 이미지를 640x640으로 크기를 조정하고 텐서로 변환합니다.
#transformer = transforms.Compose([
#    transforms.Resize((640, 640)),  # 이미지를 640x640 크기로 조정합니다.
#    transforms.ToTensor()  # 이미지를 텐서로 변환합니다.
#])

# 이미지의 최소 및 최대 크기를 설정합니다.
min_side = int(configs['min_side'])  # 이미지의 최소 측면 길이
max_side = int(configs['max_side'])  # 이미지의 최대 측면 길이

# 검증 데이터 비율을 설정합니다.
val_split = float(configs['val_split'])  # 데이터셋을 나누는 검증 비율

# 클래스 이름을 추출합니다.
if configs.get('single_column_data', False):
    unique_class_names = set()  # 클래스 이름을 저장할 집합
    for idx, row in bb_df.iterrows():
        try:
            label_data_obj = json.loads(row[configs['col_label']])  # JSON 형식의 레이블 데이터를 파싱합니다.
        except JSONDecodeError as e:
            raise Exception(f"레이블 JSON 파싱 실패: {row[configs['col_label']]}") from e
        for label in label_data_obj:
            unique_class_names.add(label['label'])  # 레이블의 이름을 집합에 추가합니다.
else:
    unique_class_names = bb_df.class_name.unique()  # 클래스 이름을 DataFrame에서 추출합니다.

# 클래스 매핑을 생성합니다.
class_mapping = misc_utils.get_cm(unique_class_names)  # 클래스 이름을 인덱스와 매핑합니다.
print(class_mapping)  # 클래스 매핑을 출력합니다.

# 역 클래스 매핑을 생성합니다.
inverse_cm = {v: k for k, v in class_mapping.items()}  # 인덱스를 클래스 이름으로 매핑합니다.
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]  # 모든 레이블 이름을 리스트로 만듭니다.

# 레이블 이름을 JSON 파일로 저장합니다.
json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))

# 데이터셋을 훈련 세트와 검증 세트로 나눕니다.
train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)

# GPU 사용에 따라 배치 크기를 설정합니다.
batch_size = gpu_opts['n_gpu'] if configs['should_use_gpu'] else 1  # GPU를 사용할 경우 GPU의 수를 배치 크기로 설정합니다.

# YOLO 데이터셋 YAML 파일 생성
image_path = op.join(images_folder.get_path(), 'images')
yaml_data = {
    'train': str(image_path),  # 이미지 폴더 경로
    'val': str(image_path),  # 검증 이미지 폴더 경로
    'nc': len(class_mapping),  # 클래스 수
    'names': [inverse_cm[i] for i in range(len(inverse_cm))]  # 클래스 이름
}

yaml_file_path = op.join(output_folder.get_path(), 'dataset.yaml')
with open(yaml_file_path, 'w') as f:
    yaml.dump(yaml_data, f)  # YAML 파일로 저장

# YOLO 형식의 어노테이션 파일 생성
labels_path = op.join(output_folder.get_path(), 'labels')
os.makedirs(labels_path, exist_ok=True)  # 라벨 저장 경로 생성

for idx, row in bb_df.iterrows():
    image_id = row['record_id']  # 이미지 ID 가져오기
    annotations = json.loads(row[configs['col_label']])  # 라벨 정보 가져오기

    # YOLO 형식으로 어노테이션 파일 준비 (class_id x_center y_center width height)
    annotation_lines = []  # 어노테이션 정보 저장 리스트
    img_path = op.join(images_path, image_id)
    img = cv2.imread(img_path)  # 이미지 파일 읽기
    h, w = img.shape[:2]  # 이미지의 높이와 너비 가져오기

    for annotation in annotations:
        cat_id = class_mapping[annotation["category"]]  # 카테고리 ID 가져오기
        xmin, ymin, bbox_width, bbox_height = annotation["bbox"]  # 바운딩 박스 좌상단 좌표 및 너비, 높이 가져오기

        # 바운딩 박스 우하단 좌표 계산
        xmax = xmin + bbox_width
        ymax = ymin + bbox_height

        # YOLO 형식으로 좌표 변환
        x_center = (xmin + bbox_width / 2) / w  # x 중심 좌표 계산
        y_center = (ymin + bbox_height / 2) / h  # y 중심 좌표 계산
        width = bbox_width / w  # 바운딩 박스 너비
        height = bbox_height / h  # 바운딩 박스 높이

        annotation_lines.append(f"{cat_id} {x_center} {y_center} {width} {height}\n")  # YOLO 형식으로 저장

    # YOLO 형식의 어노테이션을 .txt 파일로 저장
    annotation_file_path = op.join(labels_path, f"{Path(image_id).stem}.txt")
    os.makedirs(os.path.dirname(annotation_file_path), exist_ok=True)  # 경로가 없으면 생성
    with open(annotation_file_path, 'w') as f:
        f.writelines(annotation_lines)  # 어노테이션 정보 파일로 저장

# YOLOv5 모델을 훈련합니다.
model = YOLO(weights)  # 사전 학습된 가중치를 로드하여 모델을 초기화합니다.

# 모델 훈련
model.train(
    data=yaml_file_path,  # YAML 파일 경로
    epochs=int(configs.get('epochs', 1)),  # 기본값을 1으로 설정
    batch_size=batch_size,
    imgsz=min_side,  # 이미지 크기 설정
    device='cuda' if gpu_opts.get('n_gpu', 0) != 0 else 'cpu',
    project=output_path,
    name='yolov5_training'
)

# 훈련된 모델을 저장합니다.
torch.save(model.state_dict(), output_path)  # 모델의 상태 사전을 저장합니다.
