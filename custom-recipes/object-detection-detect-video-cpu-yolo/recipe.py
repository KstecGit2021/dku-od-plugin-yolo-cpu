# -*- coding: utf-8 -*-
import os.path as op  # OS 경로 관련 유틸리티 함수 모듈
import json  # JSON 데이터를 다루기 위한 모듈
import logging  # 로깅을 위한 모듈
import cv2  # OpenCV 라이브러리

import dataiku  # 데이터이쿠(Dataiku) 플랫폼과의 통신을 위한 모듈
from dataiku.customrecipe import *  # 데이터이쿠의 레시피 API 사용을 위한 모듈
import pandas as pd, numpy as np  # 데이터 처리 및 수치 연산을 위한 라이브러리
from dataiku import pandasutils as pdu  # 데이터이쿠에서 제공하는 판다스 유틸리티 함수들

from dfgenerator import DfGenerator  # 데이터프레임 제너레이터 모듈
import gpu_utils  # GPU 관련 유틸리티 함수들을 포함한 모듈
import retinanet_model  # RetinaNet 모델을 다루기 위한 모듈

# 로깅 설정: 로그 메시지를 `[Object Detection]`으로 시작하도록 포맷팅
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

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
video_out = op.join(output_folder.get_path(), 'output_video.mp4')  # 출력 비디오 파일의 경로 설정

# 사용자 지정 설정이 없으면 매 프레임마다 객체 탐지를 수행, 사용자 지정이 있으면 지정된 간격마다 탐지 수행
rate = 1 if not configs['detection_custom'] else int(configs['detection_rate'])

# 비디오 파일 열기
cap = cv2.VideoCapture(video_in)
if not cap.isOpened():
    logging.error(f"Error opening video file: {video_in}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 작성기 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

# 결과를 저장할 데이터프레임 초기화
df = pd.DataFrame(columns=['frame', 'x1', 'y1', 'x2', 'y2', 'class_name', 'confidence'])
df_idx = 0  # 데이터프레임 인덱스 초기화

# 프레임별 처리
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % rate == 0:
        # 현재 프레임에서 객체 탐지 수행
        boxes, scores, labels = retinanet_model.find_objects(model, [frame])
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score >= configs.get('confidence', 0.5):
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_name = labels_to_names[label]
                label_text = f"{label_name} {score:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 탐지된 객체 정보를 데이터프레임에 추가
                df.loc[df_idx] = [frame_count] + list(box.astype(int)) + [label_name, round(score, 2)]
                df_idx += 1
    
    # 결과 프레임을 출력 비디오에 쓰기
    out.write(frame)
    frame_count += 1

# 비디오 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()

# 성공 메시지 로깅
logging.info(f"Video processing completed. Output video saved to {video_out}")

# 바운딩 박스 결과를 출력 데이터셋에 기록
bb_ds = dataiku.Dataset(get_output_names_for_role('bboxes')[0])
bb_ds.write_with_schema(df)