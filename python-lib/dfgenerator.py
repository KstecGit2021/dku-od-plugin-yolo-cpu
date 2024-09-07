import json
from json import JSONDecodeError

import numpy as np
from keras_retinanet.preprocessing.csv_generator import CSVGenerator  # Updated import
from keras_retinanet.preprocessing.generator import Generator  # Ensure this is the correct import path

class DfGenerator(CSVGenerator):
    """Custom generator designed to work with a Pandas DataFrame in memory."""

    def __init__(self, df_data, class_mapping, cols, base_dir='', **kwargs):
        self.base_dir = base_dir  # Set base directory for image file paths
        self.cols = cols  # Set column information for DataFrame
        self.classes = class_mapping  # Set class mapping information
        self.labels = {v: k for k, v in self.classes.items()}  # Reverse class mapping to generate label information

        self.image_data = self._read_data(df_data)  # Read image data from DataFrame
        self.image_names = list(self.image_data.keys())  # Create a list of image filenames

        super(DfGenerator, self).__init__(**kwargs)  # Call parent class's initializer

    def _read_classes(self, df):
        """Reads classes from the DataFrame."""
        return {row[0]: row[1] for _, row in df.iterrows()}  # Return a dictionary with first column as key and second as value

    def __len__(self):
        """Returns the number of images."""
        return len(self.image_names)  # Return the length of the image names list

    def _read_data(self, df):
        """Reads image data and labels from the DataFrame."""
        def assert_and_retrieve(obj, prop):
            """Checks for a property in the label JSON object and retrieves it."""
            if prop not in obj:
                raise Exception(f"Property {prop} not found in label JSON object")  # Raise exception if property is missing
            return obj[prop]  # Return the property value

        data = {}
        for _, row in df.iterrows():  # Iterate over each row in the DataFrame
            img_file = row[self.cols['col_filename']]  # Get image filename
            label_data = row[self.cols['col_label']]  # Get label data
            if img_file.startswith('.') or img_file.startswith('/'):  # Remove leading '.' or '/'
                img_file = img_file[1:]

            if img_file not in data:  # Add image file to data if not present
                data[img_file] = []

            if self.cols['single_column_data']:  # If label data is in a single JSON column
                try:
                    label_data_obj = json.loads(label_data)  # Parse label data as JSON
                except JSONDecodeError as e:
                    raise Exception(f"Failed to parse label JSON: {label_data}") from e  # Raise exception on parse failure

                for label in label_data_obj:  # Iterate over each label
                    y1 = assert_and_retrieve(label, "top")  # Get top coordinate
                    x1 = assert_and_retrieve(label, "left")  # Get left coordinate
                    x2 = x1 + assert_and_retrieve(label, "width")  # Calculate right coordinate
                    y2 = y1 + assert_and_retrieve(label, "height")  # Calculate bottom coordinate
                    data[img_file].append({
                        'x1': int(x1), 'x2': int(x2),
                        'y1': int(y1), 'y2': int(y2),
                        'class': assert_and_retrieve(label, "label")  # Add label
                    })
            else:  # If label data is in separate columns
                x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]  # Get top-left coordinates
                x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]  # Get bottom-right coordinates

                # Skip images with no labels
                if not isinstance(label_data, str) and np.isnan(label_data):
                    continue

                data[img_file].append({
                    'x1': int(x1), 'x2': int(x2),
                    'y1': int(y1), 'y2': int(y2),
                    'class': label_data  # Add label
                })
        return data  # Return the collected data
