import logging
import os

import numpy as np
import cv2
import tensorflow as tf
from keras import optimizers
from keras import callbacks
from keras.utils import multi_gpu_model
from keras.models import load_model
import keras_retinanet
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.transform import random_transform_generator

import misc_utils


logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')


def get_model(weights, num_classes, freeze=False, n_gpu=None):
    """Return a RetinaNet model.

    Args:
        weights:     Initial weights.
        num_classes: Number of classes to detect.
        freeze:      Freeze the ResNet backbone.
        n_gpu:       Number of gpu, if above 1, will set up a multi gpu model.

    Returns:
        The model to save.
        The model to train.
    """
    multi_gpu = n_gpu is not None and n_gpu > 1

    modifier = freeze_model if freeze else None

    if multi_gpu:
        logging.info('Loading model in multi gpu mode.')
        with tf.device('/cpu:0'):
            model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
            model.load_weights(weights, by_name=True, skip_mismatch=True)

        multi_model = multi_gpu_model(model, gpus=n_gpu)
        return model, multi_model
    elif n_gpu == 1:
        logging.info('Loading model in single gpu mode.')
    else:
        logging.info('Loading model in cpu mode. It will be slow, use gpu if possible!')

    model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
    model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model, model


def get_test_model(weights, num_classes):
    """Returns an inference retinanet model.

    Args:
        weights:     Initial weights.
        num_classes: Number of classes to detect.
        n_gpu:       Number of gpu, if above 1, will set up a multi gpu model.

    Returns:
        The inference model.
    """
    model = get_model(weights, num_classes, freeze=True, n_gpu=1)[0]
    test_model = retinanet_bbox(model=model)
    return test_model


def compile_model(model, configs):
    """Compile retinanet."""
    if configs['optimizer'].lower() == 'adam':
        opt = optimizers.adam(lr=configs['lr'], clipnorm=0.001)
    else:
        opt = optmizers.SGD(lr=configs['lr'], momentum=True, nesterov=True, clipnorm=0.001)

    model .compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=opt
    )


def find_objects(model, paths):
    """Find objects with bach size >= 1.

    To support batch size > 1, this method implements a naive ratio grouping
    where batch of images will be processed together solely if they have a
    similar shape.

    Args:
        model: A retinanet model in inference mode.
        paths: Paths to all images to process. The *maximum* batch size is the
               number of paths.

    Returns:
        Boxes, scores, and labels.
        Their shapes: (b, 300, 4), (b, 300), (b, 300)
        With b the batch size.
    """
    if isinstance(paths, str):
        paths = [paths]

    path_i = 0
    nb_paths = len(paths)
    b_boxes, b_scores, b_labels = [], [], []

    while nb_paths != path_i:
        images = []
        scales = []
        previous_shape = None

        for path in paths[path_i:]:
            image = read_image_bgr(path)
            if previous_shape is not None and image.shape != previous_shape:
                break # Cannot make the batch bigger due to ratio difference

            previous_shape = image.shape
            path_i += 1

            image = preprocess_image(image)
            image, scale = resize_image(image)

            images.append(image)
            scales.append(scale)

        images = np.stack(images)
        boxes, scores, labels = model.predict_on_batch(images)

        for i, scale in enumerate(scales):
            boxes[i, :, :] /= scale # Taking in account the resizing factor

        b_boxes.append(boxes)
        b_scores.append(scores)
        b_labels.append(labels)



    b_boxes = np.concatenate(b_boxes, axis=0)
    b_scores = np.concatenate(b_scores, axis=0)
    b_labels = np.concatenate(b_labels, axis=0)

    return b_boxes, b_scores, b_labels


def find_objects_single(model, image, min_side=800, max_side=1333):
    """Short method to detect objects. Only supports batch size = 1."""
    if isinstance(image, str):
        image = read_image_bgr(image)
    else:
        image = image.copy()
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=min_side, max_side=max_side)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes[0, :, :] /= scale # Taking in account the resizing factor

    return boxes, scores, labels


def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """Detect objects in a video and write to the video frames.

    Args:
        model:       Trained model, must be in inference mode.
        in_vid_path: Video path.
        out_dir:     Folder where produced video will be stored.
        fps:         FPS of the produced video, it unspecified it will be the default video FPS.

    Returns:
        None
    """
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]
    out_mkv_path = os.path.join(out_dir, '{}-detected.mkv'.format(vid_name))

    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened()

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_width_height = (vid_width, vid_height)

    fps = cap.get(cv2.CAP_PROP_FPS)

    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)

    logging.info('Nb fps: {}.'.format(fps))
    nb_fps_per_min = int(fps * 60)

    idx = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        if idx % nb_fps_per_min == 0:
            logging.info('{} minutes...'.format(int(idx / fps / 60)))

        if idx % detection_rate == 0: # Detect every X frames
            boxes, scores, labels = find_objects_single(model, img)

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5: break

            misc_utils.draw_box(img, box, color=(0, 0, 255))

        vw.write(img)
        idx += 1


    cap.release()
    vw.release()

    misc_utils.mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)


def get_random_augmentator(configs):
    """Return a retinanet data augmentator. @config comes the recipe params."""
    return random_transform_generator(
        min_rotation    = float(configs['min_rotation']),
        max_rotation    = float(configs['max_rotation']),
        min_translation = (float(configs['min_trans']), float(configs['min_trans'])),
        max_translation = (float(configs['max_trans']), float(configs['max_trans'])),
        min_shear       = float(configs['min_shear']),
        max_shear       = float(configs['max_shear']),
        min_scaling     = (float(configs['min_scaling']), float(configs['min_scaling'])),
        max_scaling     = (float(configs['max_scaling']), float(configs['max_scaling'])),
        flip_x_chance   = float(configs['flip_x']),
        flip_y_chance   = float(configs['flip_y'])
    )


def get_model_checkpoint(path, base_model, n_gpu):
    if n_gpu <= 1:
        return callbacks.ModelCheckpoint(path, verbose=0, save_best_only=True, save_weights_only=True)

    return MultiGPUModelCheckpoint(path, base_model, verbose=0, save_best_only=True, save_weights_only=True)


class MultiGPUModelCheckpoint(callbacks.ModelCheckpoint):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__(filepath,
                                                      monitor=monitor,
                                                      verbose=verbose,
                                                      save_best_only=save_best_only,
                                                      save_weights_only=save_weights_only,
                                                      mode=mode,
                                                      period=period)
        self.base_model = base_model

    def on_epoch_end(self, epoch, logs=None):
        # Must behave like ModelCheckpoint on_epoch_end but save base_model instead

        # First retrieve model
        model = self.model

        # Then switching model to base model
        self.model = self.base_model

        # Calling super on_epoch_end
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # Resetting model afterwards
        self.model = model