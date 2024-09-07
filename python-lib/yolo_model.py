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

# 로깅 설정: 로그 레벨을 INFO로 설정하고, 로그 메시지 형식을 지정합니다.
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

def get_model(weights, num_classes, freeze=False, n_gpu=None):
    """Yolo 모델을 반환하는 함수.

    Args:
        weights: 초기 가중치 파일의 경로.
        num_classes: 탐지할 클래스의 수.
        freeze: ResNet 백본을 동결할지 여부.
        n_gpu: 사용할 GPU의 수, 1보다 크면 멀티 GPU 모델로 설정.

    Returns:
        모델 저장용 모델과 훈련용 모델.
    """
    multi_gpu = n_gpu is not None and n_gpu > 1  # GPU가 여러 개일 경우 multi_gpu 변수를 True로 설정

    modifier = freeze_model if freeze else None  # freeze가 True일 경우 모델 동결 함수 지정

    if multi_gpu:
        logging.info('멀티 GPU 모드에서 모델 로딩 중.')
        with tf.device('/cpu:0'):  # 모델 로딩을 CPU에서 수행
#            model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
            model = YOLO('yolov5s.pt')  # 사전 학습된 YOLOv5 작은 모델 로드
#            model = YOLO(num_classes=num_classes, modifier=modifier)
#            model.load_weights(weights, by_name=True, skip_mismatch=True)  # 가중치 로딩

        multi_model = multi_gpu_model(model, gpus=n_gpu)  # 멀티 GPU 모델 설정
        return model, multi_model
    elif n_gpu == 1:
        logging.info('싱글 GPU 모드에서 모델 로딩 중.')
    else:
        logging.info('CPU 모드에서 모델 로딩 중. 느릴 수 있습니다. 가능한 경우 GPU 사용 권장.')

#   model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
#    model.load_weights(weights, by_name=True, skip_mismatch=True)
    model = YOLO('yolov5s.pt')  # 사전 학습된 YOLOv5 작은 모델 로드

    return model, model

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

def compile_model(model, configs):
    """RetinaNet 모델을 컴파일하는 함수."""
    if configs['optimizer'].lower() == 'adam':
        opt = optimizers.adam(lr=configs['lr'], clipnorm=0.001)  # Adam 옵티마이저 설정
    else:
        opt = optimizers.SGD(lr=configs['lr'], momentum=True, nesterov=True, clipnorm=0.001)  # SGD 옵티마이저 설정

    model.compile(
        loss={
            'regression': keras_retinanet.losses.smooth_l1(),  # 회귀 손실 함수
            'classification': keras_retinanet.losses.focal()  # 분류 손실 함수
        },
        optimizer=opt
    )

def find_objects(model, paths):
    """배치 크기 >= 1인 경우 객체를 찾는 함수.

    비슷한 비율의 이미지들만 한 배치로 처리할 수 있는 간단한 배치 그룹화 방식 구현.

    Args:
        model: 추론 모드의 RetinaNet 모델.
        paths: 처리할 모든 이미지의 경로. 최대 배치 크기는 경로의 수입니다.

    Returns:
        상자, 점수 및 레이블.
        형태: (b, 300, 4), (b, 300), (b, 300)
        여기서 b는 배치 크기입니다.
    """
    if isinstance(paths, str):
        paths = [paths]  # 경로가 문자열인 경우 리스트로 변환

    path_i = 0
    nb_paths = len(paths)
    b_boxes, b_scores, b_labels = [], [], []

    while nb_paths != path_i:
        images = []
        scales = []
        previous_shape = None

        for path in paths[path_i:]:
            image = read_image_bgr(path)  # 이미지를 BGR 형식으로 읽기
            if previous_shape is not None and image.shape != previous_shape:
                break  # 비율이 다른 이미지가 포함되어 배치를 더 이상 늘릴 수 없는 경우

            previous_shape = image.shape
            path_i += 1

            image = preprocess_image(image)  # 이미지 전처리
            image, scale = resize_image(image)  # 이미지 크기 조정

            images.append(image)
            scales.append(scale)

        images = np.stack(images)
        boxes, scores, labels = model.predict_on_batch(images)  # 배치 예측 수행

        for i, scale in enumerate(scales):
            boxes[i, :, :] /= scale  # 리사이징 비율을 고려하여 상자 크기 조정

        b_boxes.append(boxes)
        b_scores.append(scores)
        b_labels.append(labels)

    b_boxes = np.concatenate(b_boxes, axis=0)
    b_scores = np.concatenate(b_scores, axis=0)
    b_labels = np.concatenate(b_labels, axis=0)

    return b_boxes, b_scores, b_labels

def find_objects_single(model, image, min_side=800, max_side=1333):
    """단일 이미지에서 객체를 탐지하는 간단한 함수. 배치 크기 = 1만 지원됨."""
    if isinstance(image, str):
        image = read_image_bgr(image)  # 이미지 파일 경로인 경우 이미지 읽기
    else:
        image = image.copy()  # 이미지가 이미 로드된 경우 복사

    image = preprocess_image(image)  # 이미지 전처리
    image, scale = resize_image(image, min_side=min_side, max_side=max_side)  # 이미지 크기 조정

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))  # 예측 수행
    boxes[0, :, :] /= scale  # 리사이징 비율을 고려하여 상자 크기 조정

    return boxes, scores, labels

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

            misc_utils.draw_box(img, box, color=(0, 0, 255))  # 상자 그리기

        vw.write(img)  # 결과 비디오에 현재 프레임 기록
        idx += 1  # 프레임 인덱스 증가

    cap.release()  # 비디오 캡처 객체 해제
    vw.release()  # 비디오 작성기 객체 해제

    # MKV 파일을 MP4로 변환하고 MKV 파일 삭제
    misc_utils.mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)

def get_random_augmentator(configs):
    """RetinaNet 데이터 증강기를 반환하는 함수.

    @config는 증강에 사용할 파라미터를 포함합니다.
    
    Args:
        configs: 증강 파라미터를 포함하는 설정 딕셔너리.

    Returns:
        데이터 증강기.
    """
    return random_transform_generator(
        min_rotation    = float(configs['min_rotation']),  # 최소 회전 각도
        max_rotation    = float(configs['max_rotation']),  # 최대 회전 각도
        min_translation = (float(configs['min_trans']), float(configs['min_trans'])),  # 최소 변환
        max_translation = (float(configs['max_trans']), float(configs['max_trans'])),  # 최대 변환
        min_shear       = float(configs['min_shear']),  # 최소 전단 변형
        max_shear       = float(configs['max_shear']),  # 최대 전단 변형
        min_scaling     = (float(configs['min_scaling']), float(configs['min_scaling'])),  # 최소 스케일링
        max_scaling     = (float(configs['max_scaling']), float(configs['max_scaling'])),  # 최대 스케일링
        flip_x_chance   = float(configs['flip_x']),  # x축 뒤집기 확률
        flip_y_chance   = float(configs['flip_y'])   # y축 뒤집기 확률
    )

def get_model_checkpoint(path, base_model, n_gpu):
    """모델 체크포인트를 반환하는 함수. 멀티 GPU 환경에서 체크포인트를 관리합니다.

    Args:
        path: 체크포인트 파일을 저장할 경로.
        base_model: 원본 모델.
        n_gpu: 사용할 GPU의 수.

    Returns:
        ModelCheckpoint 또는 MultiGPUModelCheckpoint 객체.
    """
    if n_gpu <= 1:
        return callbacks.ModelCheckpoint(path, verbose=0, save_best_only=True, save_weights_only=True)
    return MultiGPUModelCheckpoint(path, base_model, verbose=0, save_best_only=True, save_weights_only=True)

class MultiGPUModelCheckpoint(callbacks.ModelCheckpoint):
    """멀티 GPU 환경에서 모델 체크포인트를 관리하는 클래스."""

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        """초기화 메서드.

        Args:
            filepath: 체크포인트 파일 경로.
            base_model: 원본 모델.
            monitor: 모니터링할 메트릭.
            verbose: 로그 출력 수준.
            save_best_only: 가장 좋은 성능일 때만 저장 여부.
            save_weights_only: 가중치만 저장 여부.
            mode: 모니터링 모드.
            period: 체크포인트 저장 주기.
        """
        super(MultiGPUModelCheckpoint, self).__init__(filepath,
                                                      monitor=monitor,
                                                      verbose=verbose,
                                                      save_best_only=save_best_only,
                                                      save_weights_only=save_weights_only,
                                                      mode=mode,
                                                      period=period)
        self.base_model = base_model  # 원본 모델 저장

    def on_epoch_end(self, epoch, logs=None):
        """에포크 종료 시 호출되는 메서드. 원본 모델을 저장합니다.

        Args:
            epoch: 현재 에포크 번호.
            logs: 로그 정보.
        """
        # 현재 모델 가져오기
        model = self.model

        # 모델을 원본 모델로 변경
        self.model = self.base_model

        # 상위 클래스의 on_epoch_end 호출
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # 모델을 원래 상태로 복원
        self.model = model