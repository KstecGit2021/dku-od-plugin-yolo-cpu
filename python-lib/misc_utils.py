import subprocess as sp
import os
import random
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import callbacks
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import cv2
import keras_retinanet
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color


logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')


def mkv_to_mp4(mkv_path, remove_mkv=False, has_audio=True, quiet=True):
    """Transform MKV to MP4 format.

    Args:
        mkv_path:   Path to the MKV temporary file.
        remove_mkv: Delete the MKV temporary file.
        has_audio:  Keep audio in the MP4 file.
        quiet:      Silence ffmpeg conversion.

    Returns:
        None
    """
    assert os.path.isfile(mkv_path)
    print(mkv_path)
    assert os.path.splitext(mkv_path)[1] == '.mkv'
    mp4_path = os.path.splitext(mkv_path)[0] + '.mp4'

    if os.path.isfile(mp4_path):
        os.remove(mp4_path)

    #audio_codec_string = '-c:a libfdk_aac -b:a 128k' if has_audio else '-c:an'
    audio_codec_string = '-acodec copy' if has_audio else '-an'

    quiet_str = '>/dev/null 2>&1' if quiet else ''
    cmd = 'ffmpeg -i {} -vcodec copy {} {} {}'.format(
        mkv_path, audio_codec_string, mp4_path, quiet_str)

    sp.call(cmd, shell=True)


    if remove_mkv and os.path.isfile(mp4_path):
        os.remove(mkv_path) # Remove mkv only if mp4 was not created.


def split_dataset(df, val_split=0.8, shuffle=True, seed=42):
    """Split the dataset for train/val.

    Args:
        df:        Original dataframe.
        val_split: Percentage of data going to train.
        shuffle:   Shuffle the dataframe before splitting.
        seed:      Random see for reproducible split.

    Returns:
        The train dataframe.
        The validation dataframe.
    """
    paths = df.path.unique()
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(paths)

    train_paths = paths[:int(len(paths) * 0.8)]

    idxes = df.path.isin(train_paths)
    return df[idxes], df[~idxes]


def get_cm(unique_vals):
    """Returns a class mapping based on the unique values.

    The unique values must be created by Pandas' `unique()` method.

    Format: {'label_1': 0, 'label_2': 1, ...}
    """
    return {val: i for i, val in enumerate(unique_vals) if isinstance(val, str)}


def get_callbacks():
    """Returns useful callbacks."""
    return [
        callbacks.TerminateOnNaN()
    ]


def jaccard(a, b):
    """Compute the jaccard score between box a and box b."""
    side1 = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    side2 = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = side1 * side2

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union


def compute_metrics(true_pos, false_pos, false_neg):
    """Compute the precision, recall, and f1 score."""
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    if precision == 0 or recall == 0: return precision, recall, f1

    f1 = 2 / (1/precision + 1/recall)
    return precision, recall, f1


def draw_bboxes(src_path, dst_path, df, label_cap, confidence_cap, ids):
    """Draw boxes on images.

    Args:
        src_path:       Path to the source image.
        dst_path:       Path to the destination image.
        df:             Dataframe containing the bounding boxes coordinates.
        label_cap:      Boolean to add label caption on image.
        confidence_cap: Boolean to add confidence % on image.
        ids:            Ids list for each unique possible labels.

    Returns:
        None.
    """
    image = read_image_bgr(src_path)

    for _, row in df.iterrows():
        if isinstance(row.class_name, float): continue

        box = tuple(row[1:5])
        name = str(row[5])

        color = label_color(ids.index(name))

        draw_box(image, box, color=color)

        if label_cap or confidence_cap:
            txt = []
            if label_cap:
                txt = [name]
            if confidence_cap:
                confidence = round(row[6], 2)
                txt.append(str(confidence))
            draw_caption(image, box, ' '.join(txt))

    logging.info('Drawing {}'.format(dst_path))
    cv2.imwrite(dst_path, image)


def draw_caption(image, box, caption):
    """Custom version of RetinaNet `draw_caption`, to write the class name
    inside the box.

    # Arguments:
        image: The image to draw on.
        box: The bounding box of the object (x1, y1, x2, y2).
        caption: A string containing the caption.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0] + 5, b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0] + 5, b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)