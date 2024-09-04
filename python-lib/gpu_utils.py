import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def load_gpu_options(should_use_gpu, list_gpu, gpu_allocation):
    """GPU 설정을 구성합니다.
    
    Args:
        should_use_gpu: GPU를 사용할지 여부를 나타내는 불리언 값.
        list_gpu:       GPU UID를 쉼표로 구분한 문자열.
        gpu_allocation: GPU 메모리 할당 비율.
        
    Returns:
        GPU 설정을 담은 딕셔너리.
    """
    gpu_options = {}
    if should_use_gpu:
        gpu_options['n_gpu'] = len(list_gpu.split(','))  # 사용하려는 GPU의 수를 계산하여 저장
        
        config = tf.ConfigProto()  # TensorFlow 세션 설정 생성
        os.environ["CUDA_VISIBLE_DEVICES"] = list_gpu.strip()  # 사용할 GPU를 환경 변수에 설정
        config.gpu_options.visible_device_list = list_gpu.strip()  # 사용할 GPU를 TensorFlow에 설정
        config.gpu_options.per_process_gpu_memory_fraction = gpu_allocation  # GPU 메모리 할당 비율 설정
        set_session(tf.Session(config=config))  # 설정을 반영한 TensorFlow 세션 생성
    else:
        deactivate_gpu()  # GPU 비활성화 함수 호출
        gpu_options['n_gpu'] = 0  # GPU 수를 0으로 설정

    return gpu_options  # GPU 설정을 반환

def deactivate_gpu():
    """GPU를 비활성화합니다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 환경 변수를 통해 GPU를 비활성화

def can_use_gpu():
    """시스템이 GPU를 지원하는지 확인합니다."""
    # 현재 코드 환경에 'tensorflow-gpu' 패키지가 설치되어 있는지 확인
    import pip
    installed_packages = pip.get_installed_distributions()
    return "tensorflow-gpu" in [p.project_name for p in installed_packages]  # 설치된 패키지 목록에서 'tensorflow-gpu'를 검색

def set_gpus(gpus):
    """GPU 설정을 간단하게 구성하는 메서드입니다."""
    config = tf.ConfigProto()  # TensorFlow 세션 설정 생성
    config.gpu_options.visible_device_list = gpus.strip()  # 사용할 GPU를 설정
    set_session(tf.Session(config=config))  # 설정을 반영한 TensorFlow 세션 생성
