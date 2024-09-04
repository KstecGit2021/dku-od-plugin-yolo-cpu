# -*- coding: utf-8 -*-
import constants  # 상수 값을 정의한 모듈
import dataiku  # 데이터이쿠(Dataiku) 플랫폼과의 통신을 위한 모듈
import gpu_utils  # GPU 관련 유틸리티 함수들을 포함한 모듈
import json  # JSON 데이터를 다루기 위한 모듈
import logging  # 로그 출력을 위한 모듈
import misc_utils  # 다양한 유틸리티 함수들을 포함한 모듈
import numpy as np  # 수치 연산을 위한 파이썬 라이브러리
import os.path as op  # OS의 경로 관련 유틸리티 함수 모듈
import pandas as pd  # 데이터 처리 및 분석을 위한 라이브러리
import retinanet_model  # RetinaNet 모델을 다루기 위한 모듈
from dataiku import pandasutils as pdu  # 데이터이쿠에서 제공하는 판다스 유틸리티 함수들
from dataiku.customrecipe import *  # 데이터이쿠의 레시피 API 사용을 위한 모듈
from dfgenerator import DfGenerator  # 데이터프레임을 생성하는 제너레이터 모듈
from keras import callbacks  # 케라스 콜백 함수 모듈
from keras import optimizers  # 케라스 최적화 함수 모듈
from json import JSONDecodeError  # JSON 디코딩 오류 처리를 위한 예외 클래스

# 로그 설정: 레벨을 INFO로 설정하고, 로그 포맷을 정의
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

# 데이터이쿠에서 입력으로 제공되는 이미지 폴더를 가져옴
images_folder = dataiku.Folder(get_input_names_for_role('images')[0])

# 바운딩 박스 정보를 포함하는 데이터셋을 가져와 데이터프레임으로 변환
bb_df = dataiku.Dataset(get_input_names_for_role('bounding_boxes')[0]).get_dataframe()

# 가중치 파일을 저장하는 폴더를 가져옴
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정
weights = op.join(weights_folder.get_path(), 'weights.h5')

# 모델을 저장할 출력 폴더를 가져옴
output_folder = dataiku.Folder(get_output_names_for_role('model')[0])

# 모델 가중치 파일을 저장할 경로를 설정
output_path = op.join(output_folder.get_path(), 'weights.h5')

# 레시피의 설정값들을 가져옴
configs = get_recipe_config()

# GPU 사용 설정을 로드 (GPU 사용 여부, 사용할 GPU 목록, GPU 할당량 등)
gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                      configs.get('list_gpu', ''),
                                      configs.get('gpu_allocation', 0.))

# 데이터 증강을 위한 랜덤 변환 생성기를 가져옴
rnd_gen = retinanet_model.get_random_augmentator(configs)

# 이미지의 최소 및 최대 크기를 설정
min_side = int(configs['min_side'])
max_side = int(configs['max_side'])

# 검증 데이터셋 분할 비율을 설정
val_split = float(configs['val_split'])

# 단일 컬럼 데이터 여부에 따라 클래스 이름을 추출
if configs.get('single_column_data', False):
    unique_class_names = set()
    # 각 행의 라벨 정보를 JSON으로 파싱하여 클래스 이름을 추출
    for idx, row in bb_df.iterrows():
        try:
            label_data_obj = json.loads(row[configs['col_label']])
        except JSONDecodeError as e:
            raise Exception(f"Failed to parse label JSON: {row[configs['col_label']]}") from e
        for label in label_data_obj:
            unique_class_names.add(label['label'])
else:
    # 클래스 이름이 포함된 열에서 유일한 클래스 이름을 추출
    unique_class_names = bb_df.class_name.unique()

# 클래스 이름과 정수 매핑을 생성
class_mapping = misc_utils.get_cm(unique_class_names)
print(class_mapping)

# 클래스 매핑의 역매핑을 생성하여 라벨 이름 리스트를 만듦
inverse_cm = {v: k for k, v in class_mapping.items()}
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]

# 라벨 이름 리스트를 JSON 파일로 저장
json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))

# 데이터셋을 훈련/검증 세트로 분할
train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)

# GPU 사용 여부에 따라 배치 크기를 설정
if configs['should_use_gpu']:
    batch_size = gpu_opts['n_gpu']
else:
    batch_size = 1

# 훈련 데이터 제너레이터 생성
train_gen = DfGenerator(train_df, class_mapping, configs,
                        transform_generator=rnd_gen,
                        base_dir=images_folder.get_path(),
                        image_min_side=min_side,
                        image_max_side=max_side,
                        batch_size=batch_size)

# 검증 데이터 제너레이터 생성
val_gen = DfGenerator(val_df, class_mapping, configs,
                      transform_generator=None,
                      base_dir=images_folder.get_path(),
                      image_min_side=min_side,
                      image_max_side=max_side,
                      batch_size=batch_size)

# 검증 데이터가 없을 경우 None으로 설정
if len(val_gen) == 0: val_gen = None

# RetinaNet 모델을 생성하고 컴파일
model, train_model = retinanet_model.get_model(weights, len(class_mapping),
                                               freeze=configs['freeze'],
                                               n_gpu=gpu_opts['n_gpu'])

# 모델을 설정한 옵티마이저와 손실 함수로 컴파일
retinanet_model.compile_model(train_model, configs)

# 콜백 함수 목록을 설정
cbs = misc_utils.get_callbacks()

# 학습률 감소 옵션이 설정되어 있을 경우 ReduceLROnPlateau 콜백 추가
if configs.get('reducelr'):
    cbs.append(callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           patience=configs['reducelr_patience'],
                                           factor=configs['reducelr_factor'],
                                           verbose=1))

# 모델 체크포인트 콜백을 추가하여 훈련 중 최상의 모델 가중치를 저장
cbs.append(
    retinanet_model.get_model_checkpoint(output_path, model, gpu_opts['n_gpu'])
)

# 로그를 통해 훈련 시작 알림
logging.info('Training model for {} epochs.'.format(configs['epochs']))
logging.info('Nb labels: {:15}.'.format(len(class_mapping)))
logging.info('Nb images: {:15}.'.format(len(train_gen.image_names)))
logging.info('Nb val images: {:11}'.format(len(val_gen.image_names)))

# 모델 훈련 시작
train_model.fit_generator(
    train_gen,  # 훈련 데이터 제너레이터
    steps_per_epoch=len(train_gen),  # 에포크당 스텝 수
    validation_data=val_gen,  # 검증 데이터 제너레이터
    validation_steps=len(val_gen) if val_gen is not None else None,  # 검증 스텝 수
    callbacks=cbs,  # 콜백 함수들
    epochs=int(configs['epochs']),  # 훈련할 에포크 수
    verbose=2  # 훈련 과정에서의 출력 수준
)
