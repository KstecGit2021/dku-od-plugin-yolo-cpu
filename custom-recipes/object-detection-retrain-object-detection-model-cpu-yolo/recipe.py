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
from torch.utils.data import DataLoader  # 데이터 로더
from dataiku import pandasutils as pdu  # Dataiku pandas 유틸리티 모듈
from dataiku.customrecipe import *  # Dataiku 레시피 API 모듈
from dfgenerator import DfGenerator  # 사용자 정의 DataFrame 생성기
from json import JSONDecodeError  # JSON 디코딩 에러 모듈
from ultralytics import YOLO  # YOLOv5의 최신 업데이트 모듈

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

## GPU 옵션을 로드합니다. GPU 사용 여부, GPU 목록, GPU 할당량을 설정합니다.
#gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
#                                      configs.get('list_gpu', ''), 
#                                      configs.get('gpu_allocation', 0.))

# 'should_use_gpu'가 configs에 존재하지 않을 경우 기본값 False를 사용하도록 설정
should_use_gpu = configs.get('should_use_gpu', False)

# GPU 옵션을 로드합니다. GPU 사용 여부, GPU 목록, GPU 할당량을 설정합니다.
gpu_opts = gpu_utils.load_gpu_options(should_use_gpu,
                                      configs.get('list_gpu', ''), 
                                      configs.get('gpu_allocation', 0.))

# 이미지 변환을 정의합니다. 이미지를 640x640으로 크기를 조정하고 텐서로 변환합니다.
transformer = transforms.Compose([
    transforms.Resize((640, 640)),  # 이미지를 640x640 크기로 조정합니다.
    transforms.ToTensor()  # 이미지를 텐서로 변환합니다.
])

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

# 데이터 생성기를 생성합니다.
train_gen = DfGenerator(train_df, class_mapping, configs,
                        base_dir=images_folder.get_path(),
                        transform=transformer)  # 훈련 데이터에 대한 변환기 설정

val_gen = DfGenerator(val_df, class_mapping, configs,
                      base_dir=images_folder.get_path(),
                      transform=None)  # 검증 데이터에는 변환기를 사용하지 않습니다.

# 검증 데이터 생성기가 비어있으면 None으로 설정합니다.
if len(val_gen) == 0:
    val_gen = None

# PyTorch DataLoader를 생성합니다.
train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if val_gen else None

# YOLOv5 모델을 로드합니다.
model = YOLO(weights)  # 사전 학습된 가중치를 로드하여 모델을 초기화합니다.

# 훈련 파라미터를 설정합니다.
logging.info('모델을 {} 에폭 동안 훈련합니다.'.format(configs['epochs']))  # 훈련 에폭 수를 로그에 기록합니다.
logging.info('레이블 수: {:15}.'.format(len(class_mapping)))  # 레이블 수를 로그에 기록합니다.
logging.info('훈련 이미지 수: {:15}.'.format(len(train_gen.image_names)))  # 훈련 이미지 수를 로그에 기록합니다.
logging.info('검증 이미지 수: {:11}'.format(len(val_gen.image_names) if val_gen else 0))  # 검증 이미지 수를 로그에 기록합니다.

# 모델을 훈련합니다.
model.train(
    data=train_loader,
    epochs=int(configs['epochs']),  # 훈련할 에폭 수
    batch_size=batch_size,  # 배치 크기
    device='cuda' if gpu_opts['n_gpu'] !=0 else 'cpu',  # GPU를 사용할지 CPU를 사용할지 설정
    project=output_path,  # 결과를 저장할 디렉토리
    name='yolov5_training'  # 훈련 실행 이름
)

# 훈련된 모델을 저장합니다.
#torch.save(model.state_dict(), output_path)  # 모델의 상태 사전을 저장합니다.
