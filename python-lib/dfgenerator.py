import json
from json import JSONDecodeError

import numpy as np
from keras_retinanet.preprocessing.csv_generator import CSVGenerator, Generator


class DfGenerator(CSVGenerator):
    """Custom generator intented to work with in-memory Pandas' dataframe."""

    def __init__(self, df_data, class_mapping, cols, base_dir='', **kwargs):
        self.base_dir = base_dir
        self.cols = cols
        self.classes = class_mapping
        self.labels = {v: k for k, v in self.classes.items()}

        self.image_data = self._read_data(df_data)
        self.image_names = list(self.image_data.keys())

        Generator.__init__(self, **kwargs)

    def _read_classes(self, df):
        return {row[0]: row[1] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.image_names)

    def _read_data(self, df):
        def assert_and_retrieve(obj, prop):
            if prop not in obj:
                raise Exception(f"Property {prop} not found in label JSON object")
            return obj[prop]

        data = {}
        for _, row in df.iterrows():
            img_file = row[self.cols['col_filename']]
            label_data = row[self.cols['col_label']]
            if img_file[0] == '.' or img_file[0] == '/':
                img_file = img_file[1:]

            if img_file not in data:
                data[img_file] = []

            if self.cols['single_column_data']:
                try:
                    label_data_obj = json.loads(label_data)
                except JSONDecodeError as e:
                    raise Exception(f"Failed to parse label JSON: {label_data}") from e

                for label in label_data_obj:
                    y1 = assert_and_retrieve(label, "top")
                    x1 = assert_and_retrieve(label, "left")
                    x2 = assert_and_retrieve(label, "left") + assert_and_retrieve(label, "width")
                    y2 = assert_and_retrieve(label, "top") + assert_and_retrieve(label, "height")
                    data[img_file].append({
                        'x1': int(x1), 'x2': int(x2),
                        'y1': int(y1), 'y2': int(y2),
                        'class': assert_and_retrieve(label, "label")
                    })
            else:
                x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]
                x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]

                # Image without annotations
                if not isinstance(label_data, str) and np.isnan(label_data): continue

                data[img_file].append({
                    'x1': int(x1), 'x2': int(x2),
                    'y1': int(y1), 'y2': int(y2),
                    'class': label_data
                })
        return data
