# -*- coding: utf-8 -*-
import constants  # 상수를 정의하는 모듈을 가져옵니다.
import dataiku  # Dataiku 플랫폼 통합을 위한 모듈을 가져옵니다.
import gpu_utils  # GPU 관련 유틸리티 함수를 정의하는 모듈을 가져옵니다.
import json  # JSON 데이터를 처리하기 위한 모듈을 가져옵니다.
import logging  # 로깅 기능을 제공하는 모듈을 가져옵니다.
import misc_utils  # 다양한 유틸리티 함수를 제공하는 모듈을 가져옵니다.
import numpy as np  # 수치 계산을 위한 NumPy 모듈을 가져옵니다.
import os.path as op  # 파일 경로 처리를 위한 OS 모듈을 가져옵니다.
import pandas as pd  # 데이터 처리를 위한 pandas 모듈을 가져옵니다.
import torch  # PyTorch를 가져옵니다.
import torchvision.transforms as transforms  # 이미지 변환을 위한 모듈을 가져옵니다.
import yaml  # YAML 파일을 처리하기 위한 모듈을 가져옵니다.
from dataiku import pandasutils as pdu  # Dataiku와 pandas를 연결해주는 유틸리티 모듈을 가져옵니다.
from dataiku.customrecipe import *  # Dataiku의 커스텀 레시피 API를 가져옵니다.
from dfgenerator import DfGenerator  # 사용자 정의 DataFrame 생성기 모듈을 가져옵니다.
from json import JSONDecodeError  # JSON 디코딩 오류를 처리하는 모듈을 가져옵니다.
from ultralytics import YOLO  # YOLOv5 모델을 최신 업데이트된 모듈로 가져옵니다.
import cv2  # OpenCV를 가져옵니다. 이미지 처리를 위한 모듈입니다.
from pathlib import Path  # 파일 경로를 처리하기 위한 Path 모듈을 가져옵니다.

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
min_side = int(configs['min_side'])  # 이미지의 최소 측면 길이를 설정합니다.
max_side = int(configs['max_side'])  # 이미지의 최대 측면 길이를 설정합니다.

# 검증 데이터 비율을 설정합니다.
val_split = float(configs['val_split'])  # 데이터셋을 나누는 검증 비율을 설정합니다.

# 클래스 이름을 추출합니다.
if configs.get('single_column_data', False):
    unique_class_names = set()  # 클래스 이름을 저장할 집합
    for idx, row in bb_df.iterrows():
        try:
            label_data_obj = json.loads(row[configs['col_label']])  # JSON 형식의 레이블 데이터를 파싱합니다.
            # category_data_obj = label_data_obj['category']
            #for label_data in label_data_obj:
            #    category_data_obj =  label_data['category']
            #    print(category_data_obj)
        except JSONDecodeError as e:
            raise Exception(f"레이블 JSON 파싱 실패: {row[configs['col_label']]}") from e  # JSON 파싱 오류가 발생하면 예외를 발생시킵니다.
        for label in label_data_obj:
            unique_class_names.add(label['category'])  # 레이블의 카테고리 이름을 집합에 추가합니다.
else:
    unique_class_names = bb_df.class_name.unique()  # DataFrame에서 클래스 이름을 유일하게 추출합니다.

# 클래스 매핑을 생성합니다.
class_mapping = misc_utils.get_cm(unique_class_names)  # 클래스 이름을 인덱스와 매핑합니다.
# print(class_mapping)  # 클래스 매핑을 출력합니다.

# 역 클래스 매핑을 생성합니다.
inverse_cm = {v: k for k, v in class_mapping.items()}  # 인덱스를 클래스 이름으로 매핑합니다.
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]  # 모든 레이블 이름을 리스트로 만듭니다.

# 레이블 이름을 JSON 파일로 저장합니다.
json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))  # 레이블 이름을 JSON 파일로 저장합니다.

# 데이터셋을 훈련 세트와 검증 세트로 나눕니다.
#train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)
train_df, val_df = misc_utils.split_dataset(bb_df, configs['col_filename'], val_split=val_split)  # 데이터셋을 훈련 세트와 검증 세트로 분할합니다.

# GPU 사용에 따라 배치 크기를 설정합니다.
batch_size = gpu_opts['n_gpu'] if configs['should_use_gpu'] else 1  # GPU를 사용할 경우 GPU의 수를 배치 크기로 설정합니다.

images_path = op.join(images_folder.get_path())  # 이미지 폴더의 경로를 설정합니다.
# images_path

labels_path = op.join(os.path.dirname(images_path), 'labels')  # 라벨을 저장할 경로를 설정합니다.
# labels_path

# YOLO 데이터셋 YAML 파일 생성
yaml_data = {
    'train': str(images_path),  # 훈련 이미지 폴더 경로
    'val': str(images_path),  # 검증 이미지 폴더 경로
    'names': [inverse_cm[i] for i in range(len(inverse_cm))]  # 클래스 이름 리스트
}
# yaml_data

yaml_file_path = op.join(output_folder.get_path(), 'dataset.yaml')  # YAML 파일의 경로를 설정합니다.
# yaml_file_path
with open(yaml_file_path, 'w') as f:
    yaml.dump(yaml_data, f)  # YAML 파일로 데이터를 저장합니다.

# YOLO 형식의 어노테이션 파일 생성
os.makedirs(labels_path, exist_ok=True)  # 라벨 저장 경로가 없으면 생성합니다.

#print(configs['single_column_data'])
for idx, row in bb_df.iterrows():
    image_id = row[configs['col_filename']]  # 이미지 ID를 가져옵니다.

    # YOLO 형식으로 어노테이션 파일을 준비합니다. (class_id x_center y_center width height)
    annotation_lines = []  # 어노테이션 정보를 저장할 리스트를 초기화합니다.
    img_path = op.join(images_path, image_id)  # 이미지 파일 경로를 설정합니다.
    img = cv2.imread(img_path)  # 이미지를 읽어옵니다.
    h, w = img.shape[:2]  # 이미지의 높이와 너비를 가져옵니다.

    # 클래스 이름을 추출합니다.
    if configs.get('single_column_data', False):
    #if configs['single_column_data']:
        col_label_list = json.loads(row[configs['col_label']])  # JSON 형식의 라벨 정보를 가져옵니다.
        annotations = pd.DataFrame(col_label_list)  # 라벨 정보를 DataFrame으로 변환합니다.
        # for annotation in annotations:
        for index, row in annotations.iterrows():
            annotation = row.to_dict()  # 각 어노테이션을 딕셔너리로 변환합니다.
            # print(annotation)
            cat_id = class_mapping[annotation['category']]  # 카테고리 ID를 가져옵니다.
            # print(cat_id)
            xmin, ymin, bbox_width, bbox_height = annotation["bbox"]  # 바운딩 박스의 좌상단 좌표와 너비, 높이를 가져옵니다.
            # print(type(xmin))
            x_center = (xmin + bbox_width / 2) / w  # x 중심 좌표를 계산합니다.
            y_center = (ymin + bbox_height / 2) / h  # y 중심 좌표를 계산합니다.
            width = bbox_width / w  # 바운딩 박스의 너비를 계산합니다.
            height = bbox_height / h  # 바운딩 박스의 높이를 계산합니다.

            annotation_lines.append(f"{cat_id} {x_center} {y_center} {width} {height}\n")  # YOLO 형식으로 어노테이션 정보를 저장합니다.

    else:
        # 'path' 컬럼의 값이 image_id 인 행을 필터링합니다.
        annotations = bb_df[bb_df[configs['col_filename']] == image_id]  # 특정 이미지 ID에 대한 어노테이션 정보를 필터링합니다.
        # for annotation in annotations:
        for index, row in annotations.iterrows():
            annotation = row.to_dict()  # 각 어노테이션을 딕셔너리로 변환합니다.
            # print(annotation)
            cat_id = class_mapping[annotation[configs['col_label']]]  # 카테고리 ID를 가져옵니다.

            # 바운딩 박스 좌상단 좌표와 너비, 높이를 가져옵니다.
            xmin = int(annotation[configs['col_x1']])
            ymin = int(annotation[configs['col_y1']])
            xmax = int(annotation[configs['col_x2']])
            ymax = int(annotation[configs['col_y2']])

            # 바운딩 박스의 너비와 높이를 계산합니다.
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # YOLO 형식으로 좌표를 변환합니다.
            x_center = (xmin + bbox_width / 2) / w  # x 중심 좌표를 계산합니다.
            y_center = (ymin + bbox_height / 2) / h  # y 중심 좌표를 계산합니다.
            width = bbox_width / w  # 바운딩 박스의 너비를 계산합니다.
            height = bbox_height / h  # 바운딩 박스의 높이를 계산합니다.

            annotation_lines.append(f"{cat_id} {x_center} {y_center} {width} {height}\n")  # YOLO 형식으로 어노테이션 정보를 저장합니다.

    # YOLO 형식의 어노테이션을 .txt 파일로 저장합니다.
    annotation_file_path = op.join(labels_path, f"{Path(image_id).stem}.txt")  # 어노테이션 파일 경로를 설정합니다.
    os.makedirs(os.path.dirname(annotation_file_path), exist_ok=True)  # 필요한 디렉토리가 없으면 생성합니다.
    with open(annotation_file_path, 'w') as f:
        f.writelines(annotation_lines)  # YOLO 형식의 어노테이션 정보를 파일로 저장합니다.

# YOLOv5 모델을 훈련합니다.
model = YOLO(weights)  # 사전 학습된 가중치를 로드하여 YOLO 모델을 초기화합니다.

project_path = op.join(output_folder.get_path(), 'project')  # 모델 훈련 결과를 저장할 프로젝트 경로를 설정합니다.

# 모델 훈련
model.train(
    data=yaml_file_path,  # YAML 파일 경로를 설정합니다.
    epochs=int(configs.get('epochs', 1)),  # 훈련 에포크 수를 설정합니다. 기본값은 1입니다.
    batch=batch_size,  # 배치 크기를 설정합니다.
    imgsz=min_side,  # 이미지 크기를 설정합니다.
    device='cuda' if gpu_opts.get('n_gpu', 0) != 0 else 'cpu',  # GPU가 사용 가능하면 'cuda', 아니면 'cpu'를 설정합니다.
    project=project_path,  # 훈련 결과를 저장할 프로젝트 경로를 설정합니다.
    name='yolov8_training'  # 프로젝트 이름을 설정합니다.
)

# 훈련된 모델을 저장합니다.
# torch.save(model.state_dict(), output_path)  # 모델의 상태 사전을 저장하여 훈련된 모델을 저장합니다.