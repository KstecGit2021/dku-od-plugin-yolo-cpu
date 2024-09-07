# -*- coding: utf-8 -*-
import constants # 상수 값을 정의하는 모듈
import dataiku # Dataiku 플랫폼과의 통신 모듈
import gpu_utils # GPU 관련 유틸리티 함수 모듈
import json # JSON 데이터 처리 모듈
import logging # 로그 출력 모듈
import misc_utils # 다양한 유틸리티 함수 모듈
import numpy as np # 수치 연산을 위한 Python 라이브러리
import os.path as op # OS 경로 관련 유틸리티 함수 모듈
import pandas as pd # 데이터 처리 및 분석을 위한 라이브러리
import torch # YOLOv5를 위한 PyTorch
import torchvision.transforms as transforms # 이미지 변환을 위한 라이브러리
from dataiku import pandasutils as pdu # Dataiku에서 제공하는 Pandas 유틸리티 함수
from dataiku.customrecipe import * # Dataiku의 레시피 API를 사용하는 모듈
from dfgenerator import DfGenerator # 데이터 프레임을 생성하는 생성기 모듈
from json import JSONDecodeError # JSON 디코딩 오류 처리를 위한 예외 클래스
from yolov5 import YOLOv5 # YOLOv5 모듈을 불러오기 위한 모듈

# 로그 설정: 레벨을 INFO로 설정하고 로그 포맷 정의
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

# Dataiku에서 제공된 이미지 폴더를 가져옵니다.
images_folder = dataiku.Folder(get_input_names_for_role('images')[0])

# 바운딩 박스 정보가 포함된 데이터셋을 가져와서 데이터 프레임으로 변환합니다.
bb_df = dataiku.Dataset(get_input_names_for_role('bounding_boxes')[0]).get_dataframe()

# 가중치 파일을 저장할 폴더를 가져옵니다.
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정합니다.
weights = op.join(weights_folder.get_path(), 'weights.pt')  # YOLOv5는 일반적으로 .pt 파일 형식을 사용합니다.

# 모델을 저장할 출력 폴더를 가져옵니다.
output_folder = dataiku.Folder(get_output_names_for_role('model')[0])

# 모델 가중치 파일을 저장할 경로를 설정합니다.
output_path = op.join(output_folder.get_path(), 'weights.pt')

# 레시피 설정을 가져옵니다.
configs = get_recipe_config()

# GPU 사용 설정을 로드합니다 (GPU 사용 여부, 사용할 GPU 목록, GPU 할당량 등).
gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                      configs.get('list_gpu', ''),
                                      configs.get('gpu_allocation', 0.))

# 랜덤 변환 생성기를 가져옵니다 (데이터 증강을 위한).
# YOLOv5는 내장된 증강 기능을 사용하므로 전처리 조정이 필요할 수 있습니다.
transformer = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# 이미지의 최소 및 최대 크기를 설정합니다.
min_side = int(configs['min_side'])
max_side = int(configs['max_side'])

# 검증 데이터셋 분할 비율을 설정합니다.
val_split = float(configs['val_split'])

# 데이터가 단일 열 형식인지 여부에 따라 클래스 이름을 추출합니다.
if configs.get('single_column_data', False):
    unique_class_names = set()
    # 각 행의 레이블 정보를 JSON으로 파싱하여 클래스 이름을 추출합니다.
    for idx, row in bb_df.iterrows():
        try:
            label_data_obj = json.loads(row[configs['col_label']])
        except JSONDecodeError as e:
            raise Exception(f"Failed to parse label JSON: {row[configs['col_label']]}") from e
        for label in label_data_obj:
            unique_class_names.add(label['label'])
else:
    # 클래스 이름이 포함된 열에서 고유한 클래스 이름을 추출합니다.
    unique_class_names = bb_df.class_name.unique()

# 클래스 이름과 정수 매핑을 생성합니다.
class_mapping = misc_utils.get_cm(unique_class_names)
print(class_mapping)

# 클래스 매핑의 역 매핑을 생성하여 레이블 이름 목록을 만듭니다.
inverse_cm = {v: k for k, v in class_mapping.items()}
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]

# 레이블 이름 목록을 JSON 파일로 저장합니다.
json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))

# 데이터셋을 학습/검증 세트로 분할합니다.
train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)

# GPU 사용 여부에 따라 배치 크기를 설정합니다.
batch_size = gpu_opts['n_gpu'] if configs['should_use_gpu'] else 1

# 학습 데이터 생성기를 생성합니다.
train_gen = DfGenerator(train_df, class_mapping, configs,
                        transform_generator=transformer,
                        base_dir=images_folder.get_path(),
                        image_min_side=min_side,
                        image_max_side=max_side,
                        batch_size=batch_size)

# 검증 데이터 생성기를 생성합니다.
val_gen = DfGenerator(val_df, class_mapping, configs,
                      transform_generator=transformer,
                      base_dir=images_folder.get_path(),
                      image_min_side=min_side,
                      image_max_side=max_side,
                      batch_size=batch_size)

# 검증 데이터가 없으면 None으로 설정합니다.
if len(val_gen) == 0: val_gen = None

# YOLOv5 모델을 로드합니다.
# model = YOLOv5.load(weights)

model = yolo_model.get_model(weights, len(class_mapping),
                                               freeze=configs['freeze'],
                                               n_gpu=gpu_opts['n_gpu'])


# 모델을 컴파일합니다 (YOLOv5는 Keras 모델처럼 컴파일할 필요가 없습니다; 훈련할 준비가 되어 있습니다).

# 콜백 함수 목록을 설정합니다 (YOLOv5 또는 PyTorch에 맞게 조정).
# YOLOv5는 Keras 콜백을 직접 사용하지 않으므로 적절한 콜백으로 조정해야 합니다.

# 훈련 시작을 알리는 로그를 기록합니다.
logging.info('Training model for {} epochs.'.format(configs['epochs']))
logging.info('Nb labels: {:15}.'.format(len(class_mapping)))
logging.info('Nb images: {:15}.'.format(len(train_gen.image_names)))
logging.info('Nb val images: {:11}'.format(len(val_gen.image_names)))

# 모델 훈련을 시작합니다.
# YOLOv5는 Keras의 fit_generator와는 다른 훈련 루프를 사용합니다; 다음 코드는 가상의 train 메서드를 가정합니다.
train_results = model.train(
    train_loader=train_gen, 
    val_loader=val_gen,
    epochs=int(configs['epochs']),
    batch_size=batch_size,
    device='cuda' if gpu_opts['should_use_gpu'] else 'cpu'
    project=output_path,  # 결과를 저장할 디렉토리
    name='yolov5_training'  # 학습 실행 이름
)

# 준비된 dataset.yaml을 사용해 모델 학습
model.train(
    data=yaml_file_path,  # dataset.yaml 파일 경로
    epochs=1,  # 학습 에폭 수
    imgsz=640,  # 이미지 크기
    batch=1,  # 배치 크기
    project=output_folder,  # 결과를 저장할 디렉토리
    name='yolov5_training'  # 학습 실행 이름
)

# 훈련된 모델을 저장합니다.
torch.save(model.state_dict(), output_path)
