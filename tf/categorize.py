#!/usr/bin/env python3

import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt

tf.enable_eager_execution()


data_root = pathlib.Path('../img/')

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

def preprocess_image(image):
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.resize_images(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]

image = load_and_preprocess_image(image_path)
plt.imshow(image)
plt.grid(False)
plt.title(label_names[label].title())
print()
