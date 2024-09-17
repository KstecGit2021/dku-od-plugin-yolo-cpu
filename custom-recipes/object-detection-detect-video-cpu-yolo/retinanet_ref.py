다음의 retinanet 모델 기반의 동영상에서 객체를 탐지하는 로직을 yolo8 모델 기반의 객체탐지 로직으로 수정해줘

import logging
import os

import numpy as np
import cv2
import tensorflow as tf
from keras import optimizers
from keras import callbacks
from keras.utils import multi_gpu_model
from keras.models import load_model
import keras_retinanet
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.transform import random_transform_generator

import subprocess as sp
import os
import random
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow as tf
import cv2


def get_test_model(weights, num_classes):
    """추론용 RetinaNet 모델을 반환하는 함수.

    Args:
        weights: 초기 가중치 파일의 경로.
        num_classes: 탐지할 클래스의 수.

    Returns:
        추론용 모델.
    """
    model = get_model(weights, num_classes, freeze=True, n_gpu=1)[0]
    test_model = retinanet_bbox(model=model)  # RetinaNet의 bbox 추론을 위한 모델 생성
    return test_model


def draw_bboxes(src_path, dst_path, df, label_cap, confidence_cap, ids):
    """이미지에 바운딩 박스를 그립니다.

    Args:
        src_path:       원본 이미지 경로.
        dst_path:       결과 이미지 경로.
        df:             바운딩 박스 좌표를 포함한 데이터프레임.
        label_cap:      이미지에 라벨 캡션을 추가할지 여부.
        confidence_cap: 이미지에 신뢰도 %를 추가할지 여부.
        ids:            고유한 라벨에 대한 ID 리스트.

    Returns:
        None
    """
    image = read_image_bgr(src_path)  # 이미지를 BGR 형식으로 읽음

    # 데이터프레임의 각 행에 대해 바운딩 박스를 그림
    for _, row in df.iterrows():
        if isinstance(row.class_name, float): continue  # 클래스 이름이 없는 경우 건너뜀

        box = tuple(row[1:5])  # 바운딩 박스 좌표
        name = str(row[5])  # 클래스 이름

        color = label_color(ids.index(name))  # 클래스에 해당하는 색상 가져오기

        draw_box(image, box, color=color)  # 이미지에 바운딩 박스를 그림

        # 라벨 캡션이나 신뢰도를 이미지에 추가
        if label_cap or confidence_cap:
            txt = []
            if label_cap:
                txt = [name]  # 라벨 이름 추가
            if confidence_cap:
                confidence = round(row[6], 2)  # 신뢰도 추가
                txt.append(str(confidence))
            draw_caption(image, box, ' '.join(txt))  # 캡션 추가

    logging.info('Drawing {}'.format(dst_path))  # 로그 출력
    cv2.imwrite(dst_path, image)  # 결과 이미지를 저장

def mkv_to_mp4(mkv_path, remove_mkv=False, has_audio=True, quiet=True):
    """MKV 파일을 MP4 형식으로 변환합니다.

    Args:
        mkv_path:   MKV 파일의 경로.
        remove_mkv: 변환 후 MKV 파일을 삭제할지 여부.
        has_audio:  MP4 파일에 오디오를 포함할지 여부.
        quiet:      ffmpeg 변환 과정에서 메시지를 출력할지 여부.

    Returns:
        None
    """
    # MKV 파일이 존재하는지 확인
    assert os.path.isfile(mkv_path)
    print(mkv_path)
    assert os.path.splitext(mkv_path)[1] == '.mkv'  # 파일 확장자가 MKV인지 확인
    mp4_path = os.path.splitext(mkv_path)[0] + '.mp4'  # MP4 파일 경로 생성

    # 기존에 동일한 이름의 MP4 파일이 있으면 삭제
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)

    # 오디오 코덱 설정: 오디오를 유지할지 여부에 따라 달라짐
    audio_codec_string = '-acodec copy' if has_audio else '-an'

    # ffmpeg 명령어의 메시지를 숨길지 여부
    quiet_str = '>/dev/null 2>&1' if quiet else ''
    # ffmpeg 명령어 생성 및 실행
    cmd = 'ffmpeg -i {} -vcodec copy {} {} {}'.format(
        mkv_path, audio_codec_string, mp4_path, quiet_str)

    sp.call(cmd, shell=True)

    # 변환이 성공적으로 완료되면 원본 MKV 파일 삭제
    if remove_mkv and os.path.isfile(mp4_path):
        os.remove(mkv_path)

def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """비디오 파일에서 객체를 탐지하고 비디오 프레임에 결과를 기록합니다.

    Args:
        model:       훈련된 모델, 추론 모드여야 합니다.
        in_vid_path: 입력 비디오 파일 경로.
        out_dir:     결과 비디오를 저장할 폴더 경로.
        detection_rate: 객체 탐지 빈도 (프레임 단위). 지정하지 않으면 기본 비디오 FPS 사용.

    Returns:
        없음
    """
    # 비디오 파일 이름에서 확장자를 제거하고, 출력 파일 경로를 설정합니다.
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]
    out_mkv_path = os.path.join(out_dir, '{}-detected.mkv'.format(vid_name))

    # 비디오 파일을 읽기 위한 VideoCapture 객체 생성
    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened(), "비디오 파일을 열 수 없습니다."

    # 비디오 인코딩 설정
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))  # 비디오 너비
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 비디오 높이
    vid_width_height = (vid_width, vid_height)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오 FPS 정보 가져오기

    # 결과 비디오 파일을 저장하기 위한 VideoWriter 객체 생성
    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)

    logging.info('FPS: {}.'.format(fps))  # FPS 정보 로그 출력
    nb_fps_per_min = int(fps * 60)  # 1분 당 프레임 수 계산

    idx = 0
    while(cap.isOpened()):
        ret, img = cap.read()  # 비디오에서 프레임 읽기
        if not ret:
            break  # 프레임을 더 이상 읽을 수 없으면 종료

        if idx % nb_fps_per_min == 0:
            logging.info('{} 분 경과...'.format(int(idx / fps / 60)))  # 1분마다 로그 출력

        if detection_rate and idx % detection_rate == 0:  # 지정된 빈도마다 객체 탐지 수행
            boxes, scores, labels = find_objects_single(model, img)  # 객체 탐지 수행

        # 탐지된 객체의 상자, 점수, 레이블을 이미지에 그림
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5:  # 점수가 낮은 객체는 무시
                break

            draw_bboxes(img, box, color=(0, 0, 255))  # 상자 그리기

        vw.write(img)  # 결과 비디오에 현재 프레임 기록
        idx += 1  # 프레임 인덱스 증가

    cap.release()  # 비디오 캡처 객체 해제
    vw.release()  # 비디오 작성기 객체 해제

    # MKV 파일을 MP4로 변환하고 MKV 파일 삭제
    mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)


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
detect_in_video_file(model, video_in, output_folder.get_path(), detection_rate=rate)
    