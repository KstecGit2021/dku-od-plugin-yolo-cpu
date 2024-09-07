import os
import tensorflow as tf

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
        
        # 환경 변수 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = list_gpu.strip()  # 사용할 GPU를 환경 변수에 설정
        
        # TensorFlow 2.x에서 GPU 메모리 할당 비율 설정
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # GPU 메모리 할당 비율 설정 (전체 메모리의 비율을 할당)
            tf.config.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.VirtualDeviceConfiguration(memory_limit=gpu_allocation * 1024)]
            )
            
        gpu_options['n_gpu'] = len(physical_devices)  # 실제로 사용할 GPU 수 설정
    else:
        deactivate_gpu()  # GPU 비활성화 함수 호출
        gpu_options['n_gpu'] = 0  # GPU 수를 0으로 설정

    return gpu_options  # GPU 설정을 반환

def deactivate_gpu():
    """GPU를 비활성화합니다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 환경 변수를 통해 GPU를 비활성화

def can_use_gpu():
    """시스템이 GPU를 지원하는지 확인합니다."""
    # 현재 TensorFlow가 GPU를 지원하는지 확인
    return len(tf.config.list_physical_devices('GPU')) > 0  # 'GPU' 물리적 장치가 있는지 확인

def set_gpus(gpus):
    """GPU 설정을 간단하게 구성하는 메서드입니다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus.strip()  # 사용할 GPU를 환경 변수에 설정
    
    # TensorFlow 2.x에서 GPU 메모리 할당 비율 설정
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
