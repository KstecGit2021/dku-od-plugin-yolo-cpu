# -*- coding: utf-8 -*-
import os.path as op
import json

import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from dfgenerator import DfGenerator
import gpu_utils
import retinanet_model


video_folder = dataiku.Folder(get_input_names_for_role('video')[0])
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

weights = op.join(weights_folder.get_path(), 'weights.h5')
labels_to_names = json.loads(open(op.join(weights_folder.get_path(), 'labels.json')).read())

output_folder = dataiku.Folder(get_output_names_for_role('output')[0])

configs = get_recipe_config()

gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                        configs.get('list_gpu', ''),
                                        configs.get('gpu_allocation', 0.))

model = retinanet_model.get_test_model(weights, len(labels_to_names))

video_in = op.join(video_folder.get_path(), configs['video_name'])

# If using default, every frame is detected.
rate = 1 if not configs['detection_custom'] else int(configs['detection_rate'])

retinanet_model.detect_in_video_file(model, video_in, output_folder.get_path(), detection_rate=rate)
