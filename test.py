# import keras
from tensorflow import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

test_folder = "test_data"
test_output_folder = "test_data_output"
# use this to change which GPU to use
gpu = "0"

# set the modified tf session as backend in keras
setup_gpu(gpu)

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots', 'model.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Spaghetti'}

# load image

files = os.listdir(test_folder)

times = []
# This would print all the files and directories
for file in files:
    file_path = os.path.join('.', test_folder, file)

    image = read_image_bgr(file_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    start = time.time()
    image = preprocess_image(image)
    image, scale = resize_image(image)
    a = model.predict_on_batch(np.expand_dims(image, axis=0))
    print(a[0])
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    processing_time = time.time() - start
    times.append(processing_time)

    save_path = os.path.join('.', test_output_folder, f"output_{file}")
    cv2.imwrite(save_path, draw)
print(f"Average processing time: {sum(times)/len(times)}")

# plt.figure(figsize=(15, 15))
# plt.axis('off')
# plt.imshow(draw)
# plt.show()
