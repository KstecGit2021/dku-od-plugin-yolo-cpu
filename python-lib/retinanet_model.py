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

import misc_utils

# 로깅 설정: 로그의 형식과 레벨을 지정합니다.
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

def get_model(weights, num_classes, freeze=False, n_gpu=None):
    """RetinaNet 모델을 반환합니다.
    
    Args:
        weights: 초기 가중치.
        num_classes: 탐지할 클래스의 수.
        freeze: ResNet 백본을 고정할지 여부.
        n_gpu: GPU의 수. 1보다 많으면 멀티 GPU 모드로 설정.

    Returns:
        저장할 모델과 훈련할 모델.
    """
    multi_gpu = n_gpu is not None and n_gpu > 1  # 멀티 GPU 여부를 결정

    modifier = freeze_model if freeze else None  # 백본을 고정할지 여부 설정

    if multi_gpu:
        logging.info('Loading model in multi gpu mode.')
        with tf.device('/cpu:0'):  # CPU에서 모델을 로드하여 멀티 GPU에서 동작하도록 설정
            model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
            model.load_weights(weights, by_name=True, skip_mismatch=True)

        multi_model = multi_gpu_model(model, gpus=n_gpu)  # 멀티 GPU 모델로 변환
        return model, multi_model
    elif n_gpu == 1:
        logging.info('Loading model in single gpu mode.')
    else:
        logging.info('Loading model in cpu mode. It will be slow, use gpu if possible!')

    # 단일 GPU 또는 CPU 모드에서 모델을 로드
    model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
    model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model, model

def get_test_model(weights, num_classes):
    """추론용 RetinaNet 모델을 반환합니다.
    
    Args:
        weights: 초기 가중치.
        num_classes: 탐지할 클래스의 수.
    
    Returns:
        추론용 모델.
    """
    model = get_model(weights, num_classes, freeze=True, n_gpu=1)[0]  # 추론용 모델을 생성
    test_model = retinanet_bbox(model=model)  # 박스와 함께 모델을 래핑하여 추론 가능하게 함
    return test_model

def compile_model(model, configs):
    """RetinaNet 모델을 컴파일합니다."""
    if configs['optimizer'].lower() == 'adam':
        opt = optimizers.adam(lr=configs['lr'], clipnorm=0.001)  # Adam 최적화기 설정
    else:
        opt = optimizers.SGD(lr=configs['lr'], momentum=True, nesterov=True, clipnorm=0.001)  # SGD 최적화기 설정

    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),  # 회귀 손실 함수 설정
            'classification': keras_retinanet.losses.focal()  # 분류 손실 함수 설정
        },
        optimizer=opt  # 설정한 최적화기 적용
    )

def find_objects(model, paths):
    """배치 크기 >= 1일 때 객체를 찾습니다.
    
    배치 크기 > 1을 지원하기 위해, 이 메서드는 단순한 비율 그룹핑을 구현합니다. 
    유사한 모양을 가진 이미지를 하나의 배치로 처리합니다.
    
    Args:
        model: 추론 모드의 RetinaNet 모델.
        paths: 처리할 모든 이미지의 경로 목록. 최대 배치 크기는 경로의 수와 동일합니다.

    Returns:
        박스, 점수, 레이블을 반환합니다.
        이들의 형태: (b, 300, 4), (b, 300), (b, 300)으로, 여기서 b는 배치 크기입니다.
    """
    if isinstance(paths, str):
        paths = [paths]

    path_i = 0
    nb_paths = len(paths)
    b_boxes, b_scores, b_labels = [], [], []

    while nb_paths != path_i:
        images = []
        scales = []
        previous_shape = None

        for path in paths[path_i:]:
            image = read_image_bgr(path)  # 이미지를 BGR 형식으로 읽음
            if previous_shape is not None and image.shape != previous_shape:
                break  # 이미지 크기가 다르면 배치를 확장할 수 없음

            previous_shape = image.shape
            path_i += 1

            image = preprocess_image(image)  # 이미지를 전처리
            image, scale = resize_image(image)  # 이미지를 크기에 맞게 조정

            images.append(image)
            scales.append(scale)

        images = np.stack(images)  # 이미지를 스택하여 배치를 만듦
        boxes, scores, labels = model.predict_on_batch(images)  # 배치에서 예측 실행

        for i, scale in enumerate(scales):
            boxes[i, :, :] /= scale  # 크기 조정 요소 반영

        b_boxes.append(boxes)
        b_scores.append(scores)
        b_labels.append(labels)

    b_boxes = np.concatenate(b_boxes, axis=0)  # 모든 박스를 하나로 결합
    b_scores = np.concatenate(b_scores, axis=0)  # 모든 점수를 하나로 결합
    b_labels = np.concatenate(b_labels, axis=0)  # 모든 레이블을 하나로 결합

    return b_boxes, b_scores, b_labels

def find_objects_single(model, image, min_side=800, max_side=1333):
    """간단한 방법으로 객체를 탐지합니다. 배치 크기 = 1만 지원합니다."""
    if isinstance(image, str):
        image = read_image_bgr(image)  # 이미지 경로가 주어지면 이미지를 읽음
    else:
        image = image.copy()  # 이미지를 복사하여 원본을 보존
    image = preprocess_image(image)  # 이미지를 전처리
    image, scale = resize_image(image, min_side=min_side, max_side=max_side)  # 이미지 크기를 조정

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))  # 예측 실행
    boxes[0, :, :] /= scale  # 크기 조정 요소 반영

    return boxes, scores, labels

def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """비디오에서 객체를 탐지하고 이를 비디오 프레임에 기록합니다.
    
    Args:
        model: 학습된 모델, 추론 모드여야 함.
        in_vid_path: 비디오 경로.
        out_dir: 생성된 비디오가 저장될 폴더.
        detection_rate: 특정 프레임 간격마다 객체 탐지 실행.
    
    Returns:
        None
    """
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]  # 비디오 이름을 가져옴
    out_mkv_path = os.path.join(out_dir, '{}-detected.mkv'.format(vid_name))  # 출력 파일 경로 설정

    cap = cv2.VideoCapture(in_vid_path)  # 비디오 캡처 객체 생성
    assert cap.isOpened()  # 비디오 파일이 제대로 열렸는지 확인

    fourcc = cv2.VideoWriter_fourcc(*'X264')  # 비디오 코덱 설정
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))  # 비디오 너비 가져오기
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 비디오 높이 가져오기
    vid_width_height = (vid_width, vid_height)  # 비디오 크기 설정

    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오 프레임 속도 가져오기

    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)  # 비디오 쓰기 객체 생성

    logging.info('Nb fps: {}.'.format(fps))  # FPS 정보 로깅
    nb_fps_per_min = int(fps * 60)  # 분당 프레임 수 계산

    idx = 0
    while(cap.isOpened()):
        ret, img = cap.read()  # 비디오 프레임을 읽음
        if not ret:
            break

        if idx % nb_fps_per_min == 0:
            logging.info('{} minutes...'.format(int(idx / fps / 60)))  # 매 분 경과 시간 로깅

        if idx % detection_rate == 0:  # 매 detection_rate 프레임마다 객체 탐
