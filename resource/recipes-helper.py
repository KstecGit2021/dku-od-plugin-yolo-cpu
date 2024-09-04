import os
import glob

import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

import dataiku
import gpu_utils
import misc_utils
import constants

def get_dataset_info(inputs):
    """
    데이터셋의 정보를 가져옵니다.
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :return: 데이터셋의 열 정보와 GPU 사용 가능 여부를 포함하는 딕셔너리
    """
    # 'bounding_boxes' 역할을 가진 데이터셋의 이름을 가져옵니다.
    label_dataset_full_name = get_input_name_from_role(inputs, 'bounding_boxes')
    label_dataset = dataiku.Dataset(label_dataset_full_name)
    
    # 데이터셋의 열 이름을 가져옵니다.
    columns = [c['name'] for c in label_dataset.read_schema()]
    return {
        'columns': columns,  # 열 이름 리스트
        'can_use_gpu': gpu_utils.can_use_gpu()  # GPU 사용 가능 여부
    }

def has_confidence(inputs):
    """
    데이터셋에 'confidence' 열이 있는지 확인합니다.
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :return: 'confidence' 열이 존재하면 True, 그렇지 않으면 False
    """
    # 'bbox' 역할을 가진 데이터셋의 이름을 가져옵니다.
    label_dataset_full_name = get_input_name_from_role(inputs, 'bbox')
    label_dataset = dataiku.Dataset(label_dataset_full_name)
    
    # 데이터셋의 열 중 'confidence' 열이 있는지 확인합니다.
    for c in label_dataset.read_schema():
        name = c['name']
        if name == 'confidence': return True
    return False

def get_input_name_from_role(inputs, role):
    """
    주어진 역할에 해당하는 입력의 이름을 가져옵니다.
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :param role: 역할 이름 (예: 'bounding_boxes', 'bbox', 'video', 'images')
    :return: 역할에 해당하는 입력의 전체 이름
    """
    return [inp for inp in inputs if inp["role"] == role][0]['fullName']

def do(payload, config, plugin_config, inputs):
    """
    주어진 메소드에 따라 다양한 작업을 수행합니다.
    :param payload: 요청 데이터가 포함된 딕셔너리
    :param config: 구성 정보
    :param plugin_config: 플러그인 구성 정보
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :return: 요청된 작업에 따른 결과 딕셔너리
    """
    if 'method' not in payload:
        return {}

    client = dataiku.api_client()

    # 요청된 메소드에 따라 작업 수행
    if payload['method'] == 'get-dataset-info':
        response = get_dataset_info(inputs)
        response.update(get_avg_side(inputs))  # 이미지의 평균 측면 크기 추가
        return response
    if payload['method'] == 'get-gpu-info':
        return {'can_use_gpu': gpu_utils.can_use_gpu()}  # GPU 사용 가능 여부 반환
    if payload['method'] == 'get-confidence':
        return {'has_confidence': has_confidence(inputs)}  # 'confidence' 열 존재 여부 반환
    if payload['method'] == 'get-video-info':
        return {
            'can_use_gpu': gpu_utils.can_use_gpu(),  # GPU 사용 가능 여부
            'columns': get_available_videos(inputs)  # 사용 가능한 비디오 목록
        }

    return {}

def get_available_videos(inputs):
    """
    입력 역할이 'video'인 데이터셋의 사용 가능한 비디오 목록을 가져옵니다.
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :return: 비디오 파일 경로 리스트
    """
    name = get_input_name_from_role(inputs, 'video')
    folder = dataiku.Folder(name)
    
    return [f[1:] for f in folder.list_paths_in_partition()]  # 경로에서 첫 문자 슬라이스 ('/') 제거

def get_avg_side(inputs, n_first=3000):
    """
    이미지의 평균 측면 크기를 계산합니다.
    :param inputs: 데이터셋의 입력 역할 정보를 포함한 리스트
    :param n_first: 처리할 이미지의 최대 수 (기본값: 3000)
    :return: 이미지의 최소 및 최대 측면 크기 (사분위수 기반)
    """
    image_folder_full_name = get_input_name_from_role(inputs, 'images')
    image_folder = dataiku.Folder(image_folder_full_name)
    folder_path = image_folder.get_path()
    
    # 폴더 내 이미지 파일 경로를 가져오고, 처음 n_first개의 파일만 처리
    paths = image_folder.list_paths_in_partition()[:n_first]
    sides = []
    for path in paths:
        path = os.path.join(folder_path, path[1:])  # 전체 파일 경로 생성
        with Image.open(path) as img:  # PIL 라이브러리를 사용하여 이미지 열기
            w, h = img.size  # 이미지의 너비와 높이 가져오기
        sides.append(w)  # 너비 추가
        sides.append(h)  # 높이 추가
    sides = np.array(sides)  # 리스트를 NumPy 배열로 변환
    
    return {
        'min_side': int(np.percentile(sides, 25)),  # 최소 측면 크기 (1사분위수)
        'max_side': int(np.percentile(sides, 75))   # 최대 측면 크기 (3사분위수)
    }
