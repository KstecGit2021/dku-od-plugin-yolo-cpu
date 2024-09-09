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

# 로그 설정: 로그 메시지의 형식과 레벨을 설정합니다.
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

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

def split_dataset(df, val_split=0.8, shuffle=True, seed=42):
    """데이터셋을 학습/검증용으로 분할합니다.

    Args:
        df:        원본 데이터프레임.
        val_split: 학습 데이터에 사용할 비율.
        shuffle:   분할 전 데이터프레임을 섞을지 여부.
        seed:      재현 가능한 결과를 위한 랜덤 시드.

    Returns:
        학습용 데이터프레임과 검증용 데이터프레임.
    """

    # 고유한 이미지 경로 추출
    #paths = df.path.unique()
    print("aaa")
    #paths = df[col_filename].unique()
    
#    if shuffle:
#        np.random.seed(seed)
#        np.random.shuffle(paths)  # 경로를 랜덤으로 섞음

    # 학습 데이터와 검증 데이터로 분할
#    train_paths = paths[:int(len(paths) * val_split)]
#    idxes = df.path.isin(train_paths)  # 학습 데이터에 해당하는 인덱스 추출
#    return df[idxes], df[~idxes]  # 학습 및 검증 데이터프레임 반환
    return

def get_cm(unique_vals):
    """고유 값에 따른 클래스 매핑을 반환합니다.

    Pandas의 `unique()` 메서드를 통해 생성된 고유 값이어야 합니다.

    형식: {'label_1': 0, 'label_2': 1, ...}
    """
    return {val: i for i, val in enumerate(unique_vals) if isinstance(val, str)}

def get_callbacks():
    """유용한 콜백을 반환합니다."""
    return [
        callbacks.TerminateOnNaN()  # NaN 발생 시 학습을 중지하는 콜백
    ]

def jaccard(a, b):
    """박스 a와 박스 b 사이의 Jaccard 점수를 계산합니다."""
    side1 = max(0, min(a[2], b[2]) - max(a[0], b[0]))  # 교집합 영역의 너비 계산
    side2 = max(0, min(a[3], b[3]) - max(a[1], b[1]))  # 교집합 영역의 높이 계산
    inter = side1 * side2  # 교집합 영역의 면적 계산

    area_a = (a[2] - a[0]) * (a[3] - a[1])  # 박스 a의 면적
    area_b = (b[2] - b[0]) * (b[3] - b[1])  # 박스 b의 면적

    union = area_a + area_b - inter  # 합집합 영역의 면적

    return inter / union  # Jaccard 점수 반환

def compute_metrics(true_pos, false_pos, false_neg):
    """정밀도, 재현율, f1 점수를 계산합니다."""
    precision = true_pos / (true_pos + false_pos)  # 정밀도 계산
    recall = true_pos / (true_pos + false_neg)  # 재현율 계산

    if precision == 0 or recall == 0:
        f1 = 0  # F1 점수 계산에서 정밀도나 재현율이 0이면 F1 점수도 0
    else:
        f1 = 2 / (1 / precision + 1 / recall)  # F1 점수 계산

    return precision, recall, f1

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

def draw_caption(image, box, caption):
    """RetinaNet의 `draw_caption`의 커스텀 버전으로, 박스 내부에 클래스 이름을 씁니다.

    # Arguments:
        image: 캡션을 그릴 이미지.
        box:   객체의 바운딩 박스 (x1, y1, x2, y2).
        caption: 캡션으로 쓸 문자열.
    """
    b = np.array(box).astype(int)
    # 캡션을 박스 내부에 흑백으로 씀
    cv2.putText(image, caption, (b[0] + 5, b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0] + 5, b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
