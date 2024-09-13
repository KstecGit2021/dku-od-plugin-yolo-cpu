# -*- coding: utf-8 -*-
import os.path as op  # OS 경로 관련 유틸리티 함수 모듈
import json  # JSON 데이터를 다루기 위한 모듈

import dataiku  # 데이터이쿠(Dataiku) 플랫폼과의 통신을 위한 모듈
from dataiku.customrecipe import *  # 데이터이쿠의 레시피 API 사용을 위한 모듈
import pandas as pd, numpy as np  # 데이터 처리 및 수치 연산을 위한 라이브러리
from dataiku import pandasutils as pdu  # 데이터이쿠에서 제공하는 판다스 유틸리티 함수들

from dfgenerator import DfGenerator  # 데이터프레임 제너레이터 모듈
# import gpu_utils  # GPU 관련 유틸리티 함수들을 포함한 모듈
import retinanet_model  # RetinaNet 모델을 다루기 위한 모듈

# 데이터이쿠에서 입력으로 제공되는 비디오 폴더를 가져옴
video_folder = dataiku.Folder(get_input_names_for_role('video')[0])

# 데이터이쿠에서 입력으로 제공되는 가중치 폴더를 가져옴
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정
weights = op.join(weights_folder.get_path(), 'weights.h5')

# 클래스 라벨과 이름의 매핑 정보를 로드
labels_to_names = json.loads(open(op.join(weights_folder.get_path(), 'labels.json')).read())

# 데이터이쿠에서 출력으로 제공되는 폴더를 가져옴
output_folder = dataiku.Folder(get_output_names_for_role('output')[0])

# 레시피의 설정값들을 가져옴
configs = get_recipe_config()

# GPU 사용 설정을 로드 (GPU 사용 여부, 사용할 GPU 목록, GPU 할당량 등)
gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                      configs.get('list_gpu', ''),
                                      configs.get('gpu_allocation', 0.))

# 테스트용 RetinaNet 모델을 생성
model = retinanet_model.get_test_model(weights, len(labels_to_names))

# 입력 비디오 파일의 경로를 설정
video_in = op.join(video_folder.get_path(), configs['video_name'])

# 사용자 지정 설정이 없으면 매 프레임마다 객체 탐지를 수행, 사용자 지정이 있으면 지정된 간격마다 탐지 수행
rate = 1 if not configs['detection_custom'] else int(configs['detection_rate'])

# 비디오 파일에서 객체 탐지를 수행하고 결과를 출력 폴더에 저장
retinanet_model.detect_in_video_file(model, video_in, output_folder.get_path(), detection_rate=rate)
