# 이 파일은 Python 실행 가능한 create-api-service의 실제 코드입니다.
import dataiku
from dataiku.runnables import Runnable
import os
import sys
import shutil
import logging
from api_designer_utils import *

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # 서드파티 모듈의 로그를 방지하기 위한 설정
                    format='object-detection-macro %(levelname)s - %(message)s')

class MyRunnable(Runnable):
    """Python 실행 가능한 객체의 기본 인터페이스"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: 실행되는 프로젝트의 키
        :param config: 객체의 구성 딕셔너리
        :param plugin_config: 플러그인 설정을 포함
        :param client: DSS 클라이언트
        :param project: 매크로가 실행되는 DSS 프로젝트
        :param plugin_id: 사용 중인 플러그인의 이름
        """
        self.project_key = project_key  # 프로젝트 키 저장
        self.config = config  # 구성 정보 저장
        self.plugin_config = plugin_config  # 플러그인 구성 저장
        self.client = dataiku.api_client()  # DSS 클라이언트 초기화
        self.project = self.client.get_project(self.project_key)  # DSS 프로젝트 가져오기
        self.plugin_id = "object-detection-cpu-yolo"  # 플러그인 ID 설정

    def get_progress_target(self):
        """
        실행 가능한 코드가 진행 정보를 반환할 경우, 이 함수는 (target, unit) 튜플을 반환해야 합니다.
        여기서 unit은 SIZE, FILES, RECORDS, NONE 중 하나입니다.
        """
        return None  # 진행 정보를 반환하지 않음

    def run(self, progress_callback):
        """
        여기서 작업을 수행합니다. 문자열을 반환하거나 예외를 발생시킬 수 있습니다.
        progress_callback은 현재 진행 상황을 나타내는 1개의 값을 받는 함수입니다.
        """

        # 구성에서 매개변수와 DSS 클라이언트, 프로젝트 가져오기
        params = get_params(self.config, self.client, self.project)
        root_path = dataiku.get_custom_variables(project_key=self.project_key)['dip.home']

        # 플러그인을 DSS 폴더로 복사
        copy_plugin_to_dss_folder(self.plugin_id, root_path, params.get(
            "model_folder_id"), self.project_key, force_copy=True)
        # API 코드 환경 생성
        create_api_code_env(self.plugin_id, root_path, self.client, params.get(
            'code_env_name'), params.get('use_gpu'))
        # API 서비스 가져오기
        api_service = get_api_service(params, self.project)
        # 모델 엔드포인트 설정 가져오기
        endpoint_settings = get_model_endpoint_settings(params)
        # 파이썬 엔드포인트 생성
        create_python_endpoint(api_service, endpoint_settings)
        # HTML 결과 문자열 생성
        html_str = get_html_result(params)

        return html_str  # HTML 결과 문자열 반환
