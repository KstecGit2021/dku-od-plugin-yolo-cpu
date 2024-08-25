# -*- coding: utf-8 -*-
import os.path as op
import json
import logging

import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from dfgenerator import DfGenerator
import gpu_utils
import misc_utils
import retinanet_model


logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')


images_folder = dataiku.Folder(get_input_names_for_role('images')[0])
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

weights = op.join(weights_folder.get_path(), 'weights.h5')
labels_to_names = json.loads(open(op.join(weights_folder.get_path(), 'labels.json')).read())

configs = get_recipe_config()

gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                        configs.get('list_gpu', ''),
                                        configs.get('gpu_allocation', 0.))

batch_size = int(configs['batch_size'])
confidence = float(configs['confidence'])

model = retinanet_model.get_test_model(weights, len(labels_to_names))

df = pd.DataFrame(columns=['path', 'x1', 'y1', 'x2', 'y2', 'class_name', 'confidence'])
df_idx = 0

paths = images_folder.list_paths_in_partition()
folder_path = images_folder.get_path()
total_paths = len(paths)


def print_percent(i, total):
    logging.info('{}% images computed...'.format(round(100 * i / total, 2)))
    logging.info('\t{}/{}'.format(i, total))


for i in range(0, len(paths), batch_size):
    batch_paths = paths[i:i+batch_size]
    batch_paths = list(map(lambda x: op.join(folder_path, x[1:]), batch_paths))

    boxes, scores, labels = retinanet_model.find_objects(model, batch_paths)


    for batch_i in range(boxes.shape[0]):
        # For each image of the batch
        cur_path = [batch_paths[batch_i].split('/')[-1]]

        at_least_one = False
        for box, score, label in zip(boxes[batch_i], scores[batch_i], labels[batch_i]):
            if score < confidence: break # Scores are ordered.

            at_least_one = True

            int_box = list(box.astype(int))
            label_name = labels_to_names[label]

            df.loc[df_idx] = cur_path + int_box + [label_name, round(score, 2)]
            df_idx += 1

        if not at_least_one and configs['record_missing']:
            df.loc[df_idx] = cur_path + [np.nan for _ in range(6)]
            df_idx += 1

    if i % 100 == 0:
        print_percent(i, total_paths)


bb_ds = dataiku.Dataset(get_output_names_for_role('bboxes')[0])
bb_ds.write_with_schema(df)