# -*- coding: utf-8 -*-
import sys
import requests
import json
import os
import os.path as op

from dataiku.runnables import Runnable
import dataiku
import pandas as pd

import constants
import download_utils as dl_utils

class MyRunnable(Runnable):
    """Python 실행 가능한 객체의 기본 인터페이스"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: 실행되는 프로젝트의 키
        :param config: 객체의 구성 딕셔너리
        :param plugin_config: 플러그인 설정을 포함
        """
        self.project_key = project_key  # 프로젝트 키 저장
        self.config = config  # 구성 정보 저장
        self.plugin_config = plugin_config  # 플러그인 구성 저장
        self.client = dataiku.api_client()  # DSS 클라이언트 초기화

    def get_progress_target(self):
        """
        실행 가능한 코드가 진행 정보를 반환할 경우, 이 함수는 (target, unit) 튜플을 반환해야 합니다.
        여기서 unit은 SIZE, FILES, RECORDS, NONE 중 하나입니다.
        """
        return (100, 'NONE')  # 진행 정보를 반환하지 않으며, 100% 진행된 것으로 설정

    def run(self, progress_callback):
        """
        이 메소드는 실제 작업을 수행합니다. 진행 상황을 나타내는 progress_callback 함수가 제공됩니다.
        """

        # 매개변수 가져오기
        output_folder_name = self.config['folder_name']  # 출력 폴더 이름
        model = self.config['model']  # 다운로드할 모델 이름

        architecture, trained_on = model.split('_')  # 모델 이름을 아키텍처와 학습 데이터로 분리

        # 필요시 새로운 관리 폴더 생성
        project = self.client.get_project(self.project_key)  # 프로젝트 객체 가져오기

        # 지정된 폴더 이름과 일치하는 관리 폴더가 있는지 확인
        for folder in project.list_managed_folders():
            if output_folder_name == folder['name']:
                output_folder = project.get_managed_folder(folder['id'])  # 폴더가 존재하면 가져오기
                break
        else:
            # 폴더가 존재하지 않으면 새로 생성
            output_folder = project.create_managed_folder(output_folder_name)

        # 관리 폴더 경로 가져오기
        output_folder_path = dataiku.Folder(output_folder.get_definition()["id"], project_key=self.project_key).get_path()

        # 구성 파일 작성
        config = {
            "architecture": architecture,
            "trained_on": trained_on
        }

        # 레이블 다운로드
        dl_utils.download_labels(trained_on, op.join(output_folder_path, constants.LABELS_FILE))

        # S3에서 모델 가중치 다운로드 (dataiku-labs-public)
        dl_utils.download_model(architecture, trained_on,
                                op.join(output_folder_path, constants.WEIGHTS_FILE),
                                progress_callback)

        # 구성 파일을 관리 폴더에 저장
        output_folder.put_file(constants.CONFIG_FILE, json.dumps(config))

        return "<span>완료</span>"  # 작업 완료를 나타내는 HTML 문자열 반환
