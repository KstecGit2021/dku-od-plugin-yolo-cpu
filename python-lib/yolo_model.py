import logging
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from models.yolo import Model  # 가정: YOLOv5 모델이 `models.yolo`에 정의되어 있습니다.

import misc_utils

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

def get_model(weights, num_classes):
    """YOLOv5 모델을 반환하는 함수.

    Args:
        weights: 초기 가중치 파일의 경로.
        num_classes: 탐지할 클래스의 수.

    Returns:
        모델 객체.
    """
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    
    # 클래스 수 설정
    model.names = [f'class_{i}' for i in range(num_classes)]
    return model

def get_test_model(weights, num_classes):
    """추론용 YOLOv5 모델을 반환하는 함수.

    Args:
        weights: 초기 가중치 파일의 경로.
        num_classes: 탐지할 클래스의 수.

    Returns:
        추론용 모델.
    """
    return get_model(weights, num_classes)

def find_objects(model, paths):
    """배치 크기 >= 1인 경우 객체를 찾는 함수.

    Args:
        model: 추론 모드의 YOLOv5 모델.
        paths: 처리할 모든 이미지의 경로.

    Returns:
        상자, 점수 및 레이블.
        형태: (b, 300, 4), (b, 300), (b, 300)
        여기서 b는 배치 크기입니다.
    """
    if isinstance(paths, str):
        paths = [paths]  # 경로가 문자열인 경우 리스트로 변환

    b_boxes, b_scores, b_labels = [], [], []

    # 이미지 전처리를 위한 변환 정의
    transform = T.Compose([T.ToTensor()])

    for path in paths:
        image = cv2.imread(path)  # 이미지를 BGR 형식으로 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 변환
        image = transform(image).unsqueeze(0)  # 텐서로 변환하고 배치 차원 추가

        # 모델 예측
        with torch.no_grad():
            results = model(image)

        # 결과 추출
        boxes = results.xyxy[0].numpy()  # 상자 좌표
        scores = results.scores[0].numpy()  # 점수
        labels = results.names[results.pred[0][:, -1].long()].tolist()  # 레이블

        b_boxes.append(boxes)
        b_scores.append(scores)
        b_labels.append(labels)

    return np.concatenate(b_boxes, axis=0), np.concatenate(b_scores, axis=0), np.concatenate(b_labels, axis=0)

def find_objects_single(model, image, min_side=800, max_side=1333):
    """단일 이미지에서 객체를 탐지하는 간단한 함수.

    Args:
        model: 추론 모드의 YOLOv5 모델.
        image: 입력 이미지.

    Returns:
        상자, 점수, 레이블.
    """
    if isinstance(image, str):
        image = cv2.imread(image)  # 이미지 파일 경로인 경우 이미지 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 변환
    else:
        image = image.copy()  # 이미지가 이미 로드된 경우 복사

    # 이미지 크기 조정
    h, w, _ = image.shape
    scale = min(max_side / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)  # 텐서로 변환하고 배치 차원 추가

    with torch.no_grad():
        results = model(image)

    boxes = results.xyxy[0].numpy()  # 상자 좌표
    scores = results.scores[0].numpy()  # 점수
    labels = results.names[results.pred[0][:, -1].long()].tolist()  # 레이블

    return boxes, scores, labels

def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """비디오 파일에서 객체를 탐지하고 비디오 프레임에 결과를 기록합니다.

    Args:
        model:       훈련된 모델, 추론 모드여야 합니다.
        in_vid_path: 입력 비디오 파일 경로.
        out_dir:     결과 비디오를 저장할 폴더 경로.
        detection_rate: 객체 탐지 빈도 (프레임 단위).

    Returns:
        없음
    """
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]
    out_mkv_path = os.path.join(out_dir, '{}-detected.mkv'.format(vid_name))

    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened(), "비디오 파일을 열 수 없습니다."

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_width_height = (vid_width, vid_height)

    fps = cap.get(cv2.CAP_PROP_FPS)

    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)

    logging.info('FPS: {}.'.format(fps))
    nb_fps_per_min = int(fps * 60)

    idx = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        if idx % nb_fps_per_min == 0:
            logging.info('{} 분 경과...'.format(int(idx / fps / 60)))

        if detection_rate and idx % detection_rate == 0:
            boxes, scores, labels = find_objects_single(model, img)

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 상자 그리기
                cv2.putText(img, f'{label} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # 레이블과 점수 표시

        vw.write(img)
        idx += 1

    cap.release()
    vw.release()

    misc_utils.mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)

def get_random_augmentator(configs):
    """YOLOv5 데이터 증강기를 반환하는 함수.

    Args:
        configs: 증강 파라미터를 포함하는 설정 딕셔너리.

    Returns:
        데이터 증강기 (여기서는 YOLOv5의 데이터 증강은 구현되지 않음).
    """
    raise NotImplementedError("YOLOv5의 데이터 증강기는 구현되지 않았습니다.")

def get_model_checkpoint(path, base_model, n_gpu):
    """모델 체크포인트를 반환하는 함수. YOLOv5에서는 체크포인트 관리를 위한 추가 작업이 필요 없습니다.

    Args:
        path: 체크포인트 파일을 저장할 경로.
        base_model: 원본 모델.
        n_gpu: 사용할 GPU의 수.

    Returns:
        YOLOv5에서는 기본적으로 체크포인트 관리가 내장되어 있습니다.
    """
    return None
