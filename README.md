# Deep Learning Object Detection Plugin (Partially superseded)

This plugin provides tools to perform object detection using Deep Learning.

Object detection consists in detecting the location and class of several
objects in a same image.

⚠️ Starting with DSS version 10 this plugin is partially superseded by the native [object detection capabilities](https://doc.dataiku.com/dss/latest/machine-learning/computer-vision/index.html).

It comes with four recipes and a macro.


# Macro

### Download pre-trained detection yolo model

This macro downloads a pre-trained model in your project. For now only the RetinaNet
architecture pre-trained on the COCO dataset is available.

### Create scoring api service

This macro create a scoring api endpoint from the model folder.

# Recipes

### Fine-tune detection yolo model

This recipe does *transfer learning* and *finetuning* to adapt a pretrained model
on a new dataset.

There are 2 ways how you can define the inputs for the retrain recipe:
- By storing them each in a separate column: `x1`,`y1`,`x2`,`y2`,`label` and providing those column names in a recipe settings
- By storing the bounding boxes in a column as a JSON string with the following fields: `top`, `left`, `width`, `height`, `label`

### Detect objects

This recipe detects objects in images and produce a dataset storing all the
detected objects with their class and localization.

### Display bounding boxes

This recipe given objects localization, draw on images a bounding box around the
objects.

### Detect objects in video

This recipe detect objects in a video and produces a copy of the video with the
objects drawed on it.

If `ffmpeg` is installed the video will be of the mp4 format, else of the mkv format.
To install `ffmpeg`, you can use the following command:

- On Ubuntu: `sudo apt-get install ffmpeg`
- On MacOs: `brew install ffmpeg`

# References

- Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár,
[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
- Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312).

This plugin uses the implementation in Keras/Tensorflow of RetinaNet by Fizyr.
You can check the repository [here](https://github.com/fizyr/keras-retinanet), it
is under the Apache 2.0 license.
