# -*- coding: utf-8 -*-
import os.path as op
import shutil

import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import misc_utils

src_folder = dataiku.Folder(get_input_names_for_role('images')[0])
src_folder = src_folder.get_path()

dst_folder = dataiku.Folder(get_output_names_for_role('output')[0])
dst_folder = dst_folder.get_path()

configs = get_recipe_config()

label_caption = configs.get('draw_label', False)
confidence_caption = configs.get('draw_confidence', False)

bboxes = dataiku.Dataset(get_input_names_for_role('bbox')[0]).get_dataframe()

paths = bboxes.path.unique().tolist()

ids = bboxes.class_name.unique().tolist()

for path in paths:
    df = bboxes[bboxes.path == path]
    
    src_path = op.join(src_folder, path)
    dst_path = op.join(dst_folder, path)

    if len(df) == 0:
        shutil.copy(src_path, dst_path)
        continue
        
    print(path)
    misc_utils.draw_bboxes(src_path, dst_path, df, label_caption, confidence_caption, ids)
   