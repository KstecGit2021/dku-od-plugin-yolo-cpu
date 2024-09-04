"""이 모듈은 모델, 가중치 및 데이터셋을 다운로드하는 데 필요한 모든 함수를 포함하고 있습니다."""
import threading

import boto3
import botocore


def get_s3_key(archi, trained_on):
    """모델 아키텍처와 훈련된 데이터셋에 따라 S3 키를 반환합니다."""
    base = 'pretrained_models/image/object_detection'
    
    return '{}/{}_{}_weights.h5'.format(base, archi, trained_on)  # S3에서 사용할 가중치 파일의 경로 생성


def get_s3_key_labels(trained_on):
    """훈련된 데이터셋에 따라 S3 키를 반환합니다."""
    base = 'pretrained_models/image'
    
    return '{}/{}/labels.json'.format(base, trained_on)  # S3에서 사용할 라벨 파일의 경로 생성


def download_labels(trained_on, filename):
    """훈련된 데이터셋의 라벨을 다운로드합니다."""
    key = get_s3_key_labels(trained_on)  # 다운로드할 라벨 파일의 S3 키 가져오기
    
    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*', 
                                              botocore.handlers.disable_signing)  # 퍼블릭 버킷에서 다운로드를 위해 서명 비활성화
    resource.Bucket('dataiku-labs-public').download_file(key, filename)  # S3에서 로컬 파일로 다운로드


def download_model(archi, trained_on, filename, progress_callback):
    """아키텍처 @archi로 @trained_on 데이터셋에서 훈련된 모델을 @filename 경로로 다운로드합니다."""
    key = get_s3_key(archi, trained_on)  # 다운로드할 모델 파일의 S3 키 가져오기
    resource = boto3.resource('s3')
    
    # 퍼블릭 버킷에 대한 자격 증명을 비활성화합니다.
    # 버킷 정책은 아래 링크를 참조하여 업데이트해야 합니다.
    # https://docs.aws.amazon.com/AmazonS3/latest/dev/example-bucket-policies.html#example-bucket-policies-use-case-2
    resource.meta.client.meta.events.register('choose-signer.s3.*', 
                                              botocore.handlers.disable_signing)

    bucket = resource.Bucket('dataiku-labs-public')
    bucket.download_file(key, filename, 
                         Callback=ProgressTracker(resource, key, progress_callback))  # 진행 상황을 추적하면서 파일을 다운로드


def get_obj_size(resource, key):
    """S3 객체의 크기를 가져옵니다."""
    resource.meta.client.meta.events.register('choose-signer.s3.*', 
                                              botocore.handlers.disable_signing)  # 퍼블릭 버킷에서 서명 비활성화
    
    return resource.meta.client.get_object(Bucket='dataiku-labs-public', Key=key)['ContentLength']  # 객체의 크기를 반환


class ProgressTracker:
    """S3에서의 다운로드 진행 상황을 추적합니다.
    
    참고: https://stackoverflow.com/questions/41827963/track-download-progress-of-s3-file-using-boto3-and-callbacks
    """
    def __init__(self, resource, key, callback):
        self._size = get_obj_size(resource, key)  # 다운로드할 파일의 전체 크기 가져오기
        self._seen_so_far = 0  # 지금까지 다운로드된 바이트 수
        self._callback = callback  # 진행 상황을 업데이트하는 콜백 함수
        self._lock = threading.Lock()  # 쓰레드 안전성을 위한 락 설정
        
    def __call__(self, bytes_amount):
        """다운로드된 바이트 수를 업데이트하고, 진행률을 콜백 함수로 전달합니다."""
        with self._lock:
            self._seen_so_far += bytes_amount  # 다운로드된 바이트 수 증가
            percentage = (self._seen_so_far / self._size) * 100  # 다운로드된 비율 계산
            self._callback(int(percentage))  # 진행률 콜백 호출
