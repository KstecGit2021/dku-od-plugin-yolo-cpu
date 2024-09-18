# -*- coding: utf-8 -*-
import os.path as op  # OS 경로 관련 유틸리티 함수 모듈
import json  # JSON 데이터를 다루기 위한 모듈
import logging  # 로깅을 위한 모듈

import dataiku  # 데이터이쿠(Dataiku) 플랫폼과의 통신을 위한 모듈
from dataiku.customrecipe import *  # 데이터이쿠의 레시피 API 사용을 위한 모듈
import pandas as pd, numpy as np  # 데이터 처리 및 수치 연산을 위한 라이브러리
from dataiku import pandasutils as pdu  # 데이터이쿠에서 제공하는 판다스 유틸리티 함수들

from dfgenerator import DfGenerator  # 데이터프레임 제너레이터 모듈
import gpu_utils  # GPU 관련 유틸리티 함수들을 포함한 모듈
import misc_utils  # 다양한 유틸리티 함수들을 포함한 모듈
import torch  # PyTorch를 가져옵니다.
import torchvision.transforms as transforms  # 이미지 변환을 위한 모듈을 가져옵니다.
from ultralytics import YOLO  # YOLOv8 모델을 다루기 위한 모듈

# 로깅 설정: 로그 메시지를 `[Object Detection]`으로 시작하도록 포맷팅
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

# 데이터이쿠에서 입력으로 제공되는 이미지 폴더를 가져옴
images_folder = dataiku.Folder(get_input_names_for_role('images')[0])

# 데이터이쿠에서 입력으로 제공되는 가중치 폴더를 가져옴
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정
#weights = op.join(weights_folder.get_path(), 'weights.h5')
weights = op.join(weights_folder.get_path(), 'best.pt')

# 클래스 라벨과 이름의 매핑 정보를 로드
labels_to_names = json.loads(open(op.join(weights_folder.get_path(), 'labels.json')).read())

# 레시피의 설정값들을 가져옴
configs = get_recipe_config()

# GPU 사용 설정을 로드 (GPU 사용 여부, 사용할 GPU 목록, GPU 할당량 등)
gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                      configs.get('list_gpu', ''),
                                      configs.get('gpu_allocation', 0.))

# 배치 크기와 신뢰도 임계값을 설정
batch_size = int(configs['batch_size'])
confidence = float(configs['confidence'])

# 테스트용 RetinaNet 모델을 생성
model = YOLO(weights)  # 모델을 지정된 장치에 로드

# 결과를 저장할 데이터프레임을 생성
df = pd.DataFrame(columns=['path', 'x1', 'y1', 'x2', 'y2', 'class_name', 'confidence'])
df_idx = 0  # 데이터프레임 인덱스 초기화

# 이미지 폴더 내의 모든 파일 경로를 가져옴
paths = images_folder.list_paths_in_partition()
folder_path = images_folder.get_path()  # 폴더 경로를 가져옴
total_paths = len(paths)  # 전체 이미지 파일 수

# 처리 진행률을 출력하는 함수
def print_percent(i, total):
    logging.info('{}% images computed...'.format(round(100 * i / total, 2)))
    logging.info('\t{}/{}'.format(i, total))

batch_paths = []
for i in range(0, len(paths), batch_size):
    batch_path = paths[i:i + batch_size]
    batch_path = list(map(lambda x: op.join(folder_path, x[1:]), batch_path))  # 전체 경로로 변환
    batch_paths.append(batch_path)

results = []
for path in batch_paths:
    print(path[0])
    result = model.predict(path[0])
    results.append(result)

at_least_one = False
# 결과 객체에서 필요한 데이터를 추출합니다.

for result in results:
    for r in result:
        boxes = r.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표를 numpy 배열로 가져옵니다.
        conf = r.boxes.conf.cpu().numpy()  # 신뢰도 점수를 numpy 배열로 가져옵니다.
        labels = r.names  # 라벨을 numpy 배열로 가져옵니다.
        image = r.path.split('/')[-1]
        # 탐지된 각 객체에 대해 처리
        for box, conf, label in zip(boxes, conf, labels):
            if conf < confidence:  # 신뢰도가 임계값보다 낮으면 무시
                continue

            at_least_one = True  # 객체가 탐지되었음을 표시

            int_box = list(map(int, box))  # 바운딩 박스 좌표를 정수로 변환
            label_name = labels_to_names[int(label)]  # 라벨 이름을 가져옴

            # 결과를 데이터프레임에 추가
            df.loc[df_idx] = [image] + int_box + [label_name, round(conf, 2)]
            df_idx += 1

        # 탐지된 객체가 없고, 설정에 따라 누락된 경우를 기록
        if not at_least_one and configs['record_missing']:
            df.loc[df_idx] = [image] + [np.nan for _ in range(6)]  # 누락된 경우 NaN으로 채움
            df_idx += 1

        # 100번째 이미지마다 진행 상황을 출력
        if i % 100 == 0:
            print_percent(i, total_paths)

# 바운딩 박스 결과를 출력 데이터셋에 기록
bb_ds = dataiku.Dataset(get_output_names_for_role('bboxes')[0])
bb_ds.write_with_schema(df)