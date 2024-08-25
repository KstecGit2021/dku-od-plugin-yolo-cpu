# -*- coding: utf-8 -*-
import constants
import dataiku
import gpu_utils
import json
import logging
import misc_utils
import numpy as np
import os.path as op
import pandas as pd
import retinanet_model
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *
from dfgenerator import DfGenerator
from keras import callbacks
from keras import optimizers
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')


images_folder = dataiku.Folder(get_input_names_for_role('images')[0])
bb_df = dataiku.Dataset(get_input_names_for_role('bounding_boxes')[0]).get_dataframe()
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])
weights = op.join(weights_folder.get_path(), 'weights.h5')

output_folder = dataiku.Folder(get_output_names_for_role('model')[0])
output_path = op.join(output_folder.get_path(), 'weights.h5')

configs = get_recipe_config()

gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                        configs.get('list_gpu', ''),
                                        configs.get('gpu_allocation', 0.))


rnd_gen = retinanet_model.get_random_augmentator(configs)

min_side = int(configs['min_side'])
max_side = int(configs['max_side'])

val_split = float(configs['val_split'])

if configs.get('single_column_data', False):
    unique_class_names = set()
    for idx, row in bb_df.iterrows():
        try:
            label_data_obj = json.loads(row[configs['col_label']])
        except JSONDecodeError as e:
            raise Exception(f"Failed to parse label JSON: {row[configs['col_label']]}") from e
        for label in label_data_obj:
            unique_class_names.add(label['label'])
else:
    unique_class_names = bb_df.class_name.unique()
class_mapping = misc_utils.get_cm(unique_class_names)
print(class_mapping)
inverse_cm = {v: k for k, v in class_mapping.items()}
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]

json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))

train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)

if configs['should_use_gpu']:
    batch_size = gpu_opts['n_gpu']
else:
    batch_size = 1

train_gen = DfGenerator(train_df, class_mapping, configs,
                        transform_generator=rnd_gen,
                        base_dir=images_folder.get_path(),
                        image_min_side=min_side,
                        image_max_side=max_side,
                        batch_size=batch_size)

val_gen = DfGenerator(val_df, class_mapping, configs,
                      transform_generator=None,
                      base_dir=images_folder.get_path(),
                      image_min_side=min_side,
                      image_max_side=max_side,
                      batch_size=batch_size)
if len(val_gen) == 0: val_gen = None

model, train_model = retinanet_model.get_model(weights, len(class_mapping),
                                               freeze=configs['freeze'],
                                               n_gpu=gpu_opts['n_gpu'])

retinanet_model.compile_model(train_model, configs)

cbs = misc_utils.get_callbacks()
if configs.get('reducelr'):
    cbs.append(callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           patience=configs['reducelr_patience'],
                                           factor=configs['reducelr_factor'],
                                           verbose=1))

cbs.append(
    retinanet_model.get_model_checkpoint(output_path, model, gpu_opts['n_gpu'])
)


logging.info('Training model for {} epochs.'.format(configs['epochs']))
logging.info('Nb labels: {:15}.'.format(len(class_mapping)))
logging.info('Nb images: {:15}.'.format(len(train_gen.image_names)))
logging.info('Nb val images: {:11}'.format(len(val_gen.image_names)))

train_model.fit_generator(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen) if val_gen is not None else None,
    callbacks=cbs,
    epochs=int(configs['epochs']),
    verbose=2
)
