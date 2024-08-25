"""This module contains all function necessary to download models, weights, and dataset."""
import threading

import boto3
import botocore


def get_s3_key(archi, trained_on):
    """Return s3 key accordingly to the architecture and dataset it was trained on."""
    base = 'pretrained_models/image/object_detection'
    
    return '{}/{}_{}_weights.h5'.format(base, archi, trained_on)


def get_s3_key_labels(trained_on):
    """Return s3 key accordingly to dataset."""
    base = 'pretrained_models/image'
    
    return '{}/{}/labels.json'.format(base, trained_on)


def download_labels(trained_on, filename):
    """Download the labels for the dataset @trained_on."""
    key = get_s3_key_labels(trained_on)
    
    resource = boto3.resource('s3')
    resource.meta.client.meta.events.register('choose-signer.s3.*',
                                              botocore.handlers.disable_signing)
    resource.Bucket('dataiku-labs-public').download_file(key, filename)    


def download_model(archi, trained_on, filename, progress_callback):
    """Download the @archi trained on @trained_on to the path @filename."""
    key = get_s3_key(archi, trained_on)
    resource = boto3.resource('s3')
    
    # Disable the need of credentials for public bucket.
    # Note that the bucket must also have its policies updated accordingly
    # https://docs.aws.amazon.com/AmazonS3/latest/dev/example-bucket-policies.html#example-bucket-policies-use-case-2
    resource.meta.client.meta.events.register('choose-signer.s3.*',
                                              botocore.handlers.disable_signing)

    bucket = resource.Bucket('dataiku-labs-public')
    bucket.download_file(key, filename,
                         Callback=ProgressTracker(resource, key, progress_callback))


def get_obj_size(resource, key):
    resource.meta.client.meta.events.register('choose-signer.s3.*',
                                              botocore.handlers.disable_signing)
    
    return resource.meta.client.get_object(Bucket='dataiku-labs-public', Key=key)['ContentLength']
   

class ProgressTracker:
    """Track the download from s3.
    
    https://stackoverflow.com/questions/41827963/track-download-progress-of-s3-file-using-boto3-and-callbacks
    """
    def __init__(self, resource, key, callback):
        self._size = get_obj_size(resource, key)
        self._seen_so_far = 0
        self._callback = callback
        self._lock = threading.Lock()
        
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            self._callback(int(percentage))
