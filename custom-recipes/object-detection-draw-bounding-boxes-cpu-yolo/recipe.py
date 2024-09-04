# -*- coding: utf-8 -*-
import os.path as op  # OS 경로 관련 유틸리티 함수 모듈
import shutil  # 파일 및 디렉토리 작업을 위한 고급 유틸리티 모듈

import dataiku  # 데이터이쿠(Dataiku) 플랫폼과의 통신을 위한 모듈
from dataiku.customrecipe import *  # 데이터이쿠의 레시피 API 사용을 위한 모듈
import pandas as pd, numpy as np  # 데이터 처리 및 수치 연산을 위한 라이브러리
from dataiku import pandasutils as pdu  # 데이터이쿠에서 제공하는 판다스 유틸리티 함수들

import misc_utils  # 다양한 유틸리티 함수들을 포함한 모듈

# 데이터이쿠에서 입력으로 제공되는 이미지 폴더를 가져와 경로를 설정
src_folder = dataiku.Folder(get_input_names_for_role('images')[0])
src_folder = src_folder.get_path()

# 데이터이쿠에서 출력으로 제공되는 폴더를 가져와 경로를 설정
dst_folder = dataiku.Folder(get_output_names_for_role('output')[0])
dst_folder = dst_folder.get_path()

# 레시피의 설정값들을 가져옴
configs = get_recipe_config()

# 바운딩 박스 위에 라벨을 그릴지 여부를 설정
label_caption = configs.get('draw_label', False)

# 바운딩 박스 위에 신뢰도(Confidence) 값을 그릴지 여부를 설정
confidence_caption = configs.get('draw_confidence', False)

# 바운딩 박스 정보가 포함된 데이터셋을 가져와 데이터프레임으로 변환
bboxes = dataiku.Dataset(get_input_names_for_role('bbox')[0]).get_dataframe()

# 바운딩 박스 정보에서 이미지 경로들의 유일한 목록을 추출
paths = bboxes.path.unique().tolist()

# 바운딩 박스 정보에서 클래스 이름들의 유일한 목록을 추출
ids = bboxes.class_name.unique().tolist()

# 각 이미지 경로에 대해 처리 시작
for path in paths:
    # 현재 경로에 해당하는 바운딩 박스 정보를 필터링
    df = bboxes[bboxes.path == path]
    
    # 소스 이미지 파일의 전체 경로를 설정
    src_path = op.join(src_folder, path)
    
    # 대상 이미지 파일의 전체 경로를 설정
    dst_path = op.join(dst_folder, path)

    # 현재 이미지에 대한 바운딩 박스 정보가 없을 경우, 원본 이미지를 복사
    if len(df) == 0:
        shutil.copy(src_path, dst_path)
        continue
        
    # 현재 처리 중인 이미지 경로를 출력
    print(path)
    
    # 바운딩 박스를 이미지에 그려서 저장
    misc_utils.draw_bboxes(src_path, dst_path, df, label_caption, confidence_caption, ids)
