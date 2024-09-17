import logging
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import subprocess as sp

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def load_yolo_model(weights_path):
    """YOLOv8 모델을 로드하는 함수.
    
    Args:
        weights_path: YOLOv8 가중치 파일의 경로.

    Returns:
        YOLOv8 모델.
    """
    model = YOLO(weights_path)  # YOLOv8 모델 로드
    return model

def draw_bboxes(img, boxes, scores, labels, confidence_cap):
    """이미지에 바운딩 박스를 그립니다.

    Args:
        img:            원본 이미지.
        boxes:          바운딩 박스 좌표.
        scores:         객체의 신뢰도 점수.
        labels:         객체의 클래스 레이블.
        confidence_cap: 신뢰도 캡션을 추가할지 여부.

    Returns:
        None
    """
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5:  # 점수가 낮은 객체는 무시
            continue
        
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # 상자 색상 설정 (초록색)

        # 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if confidence_cap:
            caption = f"{label} {score:.2f}"
            cv2.putText(img, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img

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
    assert os.path.isfile(mkv_path)
    mp4_path = os.path.splitext(mkv_path)[0] + '.mp4'

    if os.path.isfile(mp4_path):
        os.remove(mp4_path)

    audio_codec_string = '-acodec copy' if has_audio else '-an'
    quiet_str = '>/dev/null 2>&1' if quiet else ''
    cmd = f'ffmpeg -i {mkv_path} -vcodec copy {audio_codec_string} {mp4_path} {quiet_str}'

    sp.call(cmd, shell=True)

    if remove_mkv and os.path.isfile(mp4_path):
        os.remove(mkv_path)

def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """비디오 파일에서 객체를 탐지하고 비디오 프레임에 결과를 기록합니다.

    Args:
        model:       YOLOv8 모델.
        in_vid_path: 입력 비디오 파일 경로.
        out_dir:     결과 비디오를 저장할 폴더 경로.
        detection_rate: 객체 탐지 빈도 (프레임 단위). 지정하지 않으면 기본 비디오 FPS 사용.

    Returns:
        None
    """
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]
    out_mkv_path = os.path.join(out_dir, f'{vid_name}-detected.mkv')

    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened(), "비디오 파일을 열 수 없습니다."

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_width_height = (vid_width, vid_height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)

    logging.info(f'FPS: {fps}.')
    nb_fps_per_min = int(fps * 60)

    idx = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        if idx % nb_fps_per_min == 0:
            logging.info(f'{int(idx / fps / 60)} 분 경과...')

        if detection_rate and idx % detection_rate == 0:
            results = model(img)
            boxes = results.xyxy[0][:, :4].cpu().numpy()  # [x1, y1, x2, y2]
            scores = results.xyxy[0][:, 4].cpu().numpy()  # 신뢰도
            labels = results.xyxy[0][:, 5].astype(int).cpu().numpy()  # 클래스 레이블

            img = draw_bboxes(img, boxes, scores, labels, confidence_cap=True)

        vw.write(img)
        idx += 1

    cap.release()
    vw.release()

    mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)

# 메인 부분
if __name__ == '__main__':
    weights_path = 'path/to/yolov8_weights.pt'  # YOLOv8 가중치 파일 경로
    model = load_yolo_model(weights_path)

    input_video_path = 'path/to/input_video.mp4'
    output_folder = 'path/to/output_folder'
    detection_rate = 1  # 매 프레임마다 객체 탐지

    detect_in_video_file(model, input_video_path, output_folder, detection_rate)
