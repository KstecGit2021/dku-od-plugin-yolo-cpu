# -*- coding: utf-8 -*-
import constants  # Define constants
import dataiku  # Dataiku platform integration
import gpu_utils  # GPU utility functions
import json  # JSON handling
import logging  # Logging
import misc_utils  # Utility functions
import numpy as np  # Numerical operations
import os.path as op  # OS path utilities
import pandas as pd  # Data handling
import torch  # PyTorch
import torchvision.transforms as transforms  # Image transformations
from dataiku import pandasutils as pdu  # Dataiku pandas utilities
from dataiku.customrecipe import *  # Dataiku recipe API
from dfgenerator import DfGenerator  # Custom DataFrame generator
from json import JSONDecodeError  # JSON decode error
from yolov5 import YOLOv5  # YOLOv5 for object detection

# Configure logging
logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')

# Get images folder from Dataiku
images_folder = dataiku.Folder(get_input_names_for_role('images')[0])

# Load bounding boxes dataset into DataFrame
bb_df = dataiku.Dataset(get_input_names_for_role('bounding_boxes')[0]).get_dataframe()

# Get weights folder from Dataiku
weights_folder = dataiku.Folder(get_input_names_for_role('weights')[0])

# Set weights file path
weights = op.join(weights_folder.get_path(), 'weights.pt')  # YOLOv5 uses .pt files

# Get output folder from Dataiku
output_folder = dataiku.Folder(get_output_names_for_role('model')[0])

# Set output path for model weights
output_path = op.join(output_folder.get_path(), 'weights.pt')

# Load recipe configuration
configs = get_recipe_config()

# Load GPU options
gpu_opts = gpu_utils.load_gpu_options(configs.get('should_use_gpu', False),
                                      configs.get('list_gpu', ''),
                                      configs.get('gpu_allocation', 0.))

# Define image transformations
transformer = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Set minimum and maximum image sizes
min_side = int(configs['min_side'])
max_side = int(configs['max_side'])

# Set validation split ratio
val_split = float(configs['val_split'])

# Extract class names
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

# Create class mapping
class_mapping = misc_utils.get_cm(unique_class_names)
print(class_mapping)

# Create inverse class mapping
inverse_cm = {v: k for k, v in class_mapping.items()}
labels_names = [inverse_cm[i] for i in range(len(inverse_cm))]

# Save label names to JSON file
json.dump(labels_names, open(op.join(output_folder.get_path(), constants.LABELS_FILE), 'w+'))

# Split dataset into training and validation sets
train_df, val_df = misc_utils.split_dataset(bb_df, val_split=val_split)

# Set batch size based on GPU usage
batch_size = gpu_opts['n_gpu'] if configs['should_use_gpu'] else 1

# Create data generators
train_gen = DfGenerator(train_df, class_mapping, configs,
                        transform_generator=transformer,
                        base_dir=images_folder.get_path(),
                        image_min_side=min_side,
                        image_max_side=max_side,
                        batch_size=batch_size)

val_gen = DfGenerator(val_df, class_mapping, configs,
                      transform_generator=transformer,
                      base_dir=images_folder.get_path(),
                      image_min_side=min_side,
                      image_max_side=max_side,
                      batch_size=batch_size)

if len(val_gen) == 0:
    val_gen = None

# Load YOLOv5 model
model = YOLOv5.load(weights)

# Set training parameters
logging.info('Training model for {} epochs.'.format(configs['epochs']))
logging.info('Nb labels: {:15}.'.format(len(class_mapping)))
logging.info('Nb images: {:15}.'.format(len(train_gen.image_names)))
logging.info('Nb val images: {:11}'.format(len(val_gen.image_names)))

# Train the model
model.train(
    train_loader=train_gen, 
    val_loader=val_gen,
    epochs=int(configs['epochs']),
    batch_size=batch_size,
    device='cuda' if gpu_opts['should_use_gpu'] else 'cpu',
    project=output_path,  # Directory to save results
    name='yolov5_training'  # Training run name
)

# Save the trained model
#torch.save(model.state_dict(), output_path)
