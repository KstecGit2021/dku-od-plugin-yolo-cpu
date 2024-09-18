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
import ultralytics  # YOLOv8 모델을 다루기 위한 모듈
from PIL import Image  # 이미지 처리 라이브러리
import torch  # PyTorch 라이브러리
import cv2  # OpenCV 라이브러리

# 데이터이쿠에서 입력으로 제공되는 비디오 폴더를 가져옴
video_folder = dataiku.Folder(get_input_names_for_role('video')[0])

# 데이터이쿠에서 입력으로 제공되는 가중치 폴더를 가져옴
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# 가중치 파일의 경로를 설정
weights = op.join(weights_folder.get_path(), 'best.pt')

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

def get_model(weights, n_gpu=None):
    """YOLOv8 모델을 반환하는 함수.

    Args:
        weights: 초기 가중치 파일의 경로.
        n_gpu: 사용할 GPU의 수, 1보다 크면 멀티 GPU 모델로 설정.

    Returns:
        모델 저장용 모델과 훈련용 모델.
    """
    # 사용할 장치를 설정 (CUDA가 사용 가능하고 n_gpu가 0보다 크면 GPU 사용)
    device = torch.device('cuda' if torch.cuda.is_available() and n_gpu > 0 else 'cpu')

    # 가중치 로딩
    if weights is not None:
        logging.info('모델 가중치를 로딩 중: %s', weights)
        # YOLOv8 모델 로드
        model = ultralytics.YOLO(weights)

    # 다중 GPU 설정
    if n_gpu is not None and n_gpu > 1:
        logging.info('멀티 GPU 모드에서 모델 로딩 중.')
        model = torch.nn.DataParallel(model)  # 데이터 병렬 모델로 변환
        model = model.to(device)  # GPU로 모델 전송
    elif n_gpu == 1:
        logging.info('싱글 GPU 모드에서 모델 로딩 중.')
        model = model.to(device)  # GPU로 모델 전송
    else:
        logging.info('CPU 모드에서 모델 로딩 중. 느릴 수 있습니다. 가능한 경우 GPU 사용 권장.')
        # CPU 모드에서는 특별한 설정 필요 없음

    return model  # 로드된 모델 반환

# YOLOv8 모델을 생성
model = get_model(weights, gpu_opts['n_gpu'])

# 입력 비디오 파일의 경로를 설정
video_in = op.join(video_folder.get_path(), configs['video_name'])

# 사용자 지정 설정이 없으면 매 프레임마다 객체 탐지를 수행, 사용자 지정이 있으면 지정된 간격마다 탐지 수행
rate = 1 if not configs['detection_custom'] else int(configs['detection_rate'])

def detect_in_video_file(model, video_in, output_path, name, rate=1):
    """
    비디오 파일에서 특정 프레임 비율로 객체 탐지를 수행하고 결과를 출력 폴더에 저장
    결과 프레임들을 JPEG 파일로 저장하고 MP4 비디오 파일로 저장

    :param model: YOLOv8 모델 인스턴스
    :param video_in: 입력 비디오 파일 경로
    :param output_path: 결과를 저장할 출력 경로
    :param name: 출력 비디오 파일 이름 및 하위 폴더 이름
    :param rate: 탐지할 프레임 비율 (예: 1이면 모든 프레임, 2이면 2프레임마다 탐지)
    """
    # 출력 폴더 경로 생성
    full_output_path = os.path.join(output_path, name)

    # JPEG 및 MP4 하위 폴더 경로 생성
    jpg_output_path = os.path.join(full_output_path, 'jpg')
    mp4_output_path = os.path.join(full_output_path, 'mp4')

    # 출력 폴더가 없으면 생성
    os.makedirs(jpg_output_path, exist_ok=True)  # JPEG 파일을 저장할 폴더 생성
    os.makedirs(mp4_output_path, exist_ok=True)  # MP4 비디오를 저장할 폴더 생성

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_in)  # 입력 비디오 파일 열기
    if not cap.isOpened():  # 비디오 파일 열기 실패 시
        raise ValueError("비디오 파일을 열 수 없습니다.")

    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 비디오 프레임 너비
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 비디오 프레임 높이

    # YOLOv8 결과를 저장할 리스트
    results = []

    # MP4 비디오 파일 생성
    output_video_file = os.path.join(mp4_output_path, f"{name}")  # 출력 비디오 파일 경로
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec 설정
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))  # 비디오 작성 객체 생성

    frame_idx = 0  # 프레임 인덱스 초기화
    while True:
        ret, frame = cap.read()  # 비디오에서 프레임 읽기
        if not ret:  # 프레임을 읽지 못한 경우 루프 종료
            break

        # 지정된 프레임 비율에 따라 객체 탐지
        if frame_idx % rate == 0:  # 설정된 프레임 간격에 따라 탐지
            # YOLO 모델로 객체 탐지 수행
            result = model(frame)  # 현재 프레임에 대해 객체 탐지

            # `result`는 리스트로 반환되므로 각 항목에 대해 처리
            for res in result:
                # 탐지 결과를 비디오에 그리기
                annotated_frame = res.plot()  # plot 메서드를 사용하여 결과 그리기

                # 프레임을 JPEG 파일로 저장
                jpeg_file = os.path.join(jpg_output_path, f"frame_{frame_idx}.jpg")  # JPEG 파일 경로
                cv2.imwrite(jpeg_file, annotated_frame)  # JPEG 파일로 저장

                # 프레임을 MP4 비디오에 추가
                out.write(annotated_frame)  # 비디오에 프레임 추가

            results.extend(result)  # 모든 탐지 결과를 리스트에 추가

        frame_idx += 1  # 프레임 인덱스 증가

    # 비디오 캡처 객체와 비디오 작성 객체 해제
    cap.release()  # 비디오 캡처 객체 해제
    out.release()  # 비디오 작성 객체 해제
    cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

    return results  # 탐지 결과 반환

# 비디오 파일에서 객체 탐지 수행
results = detect_in_video_file(model, video_in, output_folder.get_path(), configs['video_name'], rate)

def save_results(results_list, base_path):
    """
    결과 리스트를 받아서 base_path에 메타데이터와 이미지를 저장합니다.

    :param results_list: 객체 탐지 결과 리스트 (각 결과는 특정 형식의 객체)
    :param base_path: 결과를 저장할 기본 경로
    """
    # base_path가 존재하지 않으면 생성
    if not os.path.exists(base_path):
        os.makedirs(base_path)  # 지정된 경로 생성

    for i, result in enumerate(results_list):  # 각 결과에 대해 반복
        result_data = {
            'names': result.names,  # 탐지된 객체의 이름
            'orig_shape': result.orig_shape,  # 원본 이미지의 크기
            'path': result.path,  # 이미지 경로
            'save_dir': result.save_dir,  # 저장 디렉토리
            'speed': result.speed,  # 탐지 속도
            # 이미지 배열은 별도로 저장됨
        }

        # 메타데이터 저장
        json_file_path = os.path.join(base_path, f'result_{i}.json')  # JSON 파일 경로
        with open(json_file_path, 'w') as f:
            json.dump(result_data, f)  # JSON 파일로 결과 저장

        # 이미지 배열을 PNG 파일로 저장
        img = Image.fromarray(result.orig_img)  # 이미지 배열을 PIL 이미지로 변환
        image_file_path = os.path.join(base_path, f'result_{i}_image.png')  # 이미지 파일 경로
        img.save(image_file_path)  # PNG 파일로 저장

# 탐지 결과를 지정된 경로에 저장
save_results(results, os.path.join(output_folder.get_path(), configs['video_name'], "result"))
