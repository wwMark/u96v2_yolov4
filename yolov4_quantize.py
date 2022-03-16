import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# quantize
mnist_test, info_test = tfds.load('mnist', split='test', shuffle_files=True, with_info=True)
mnist_test = mnist_test.take(256)

# create numpy array
images_array = np.empty((0, 608, 608, 3), float)
for example in mnist_test.as_numpy_iterator():
    image = example['image']
    image = np.stack((image)*3, axis=-1)
    image = Image.fromarray(image, 'RGB')
    resized_image = image.resize((608, 608))
    image_data_array = np.asarray(resized_image)
    image_data_array = np.expand_dims(image_data_array, axis=0)
    images_array = np.append(images_array, image_data_array, axis=0)

model = tf.keras.models.load_model('./yolov4_save')
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=images_array)

quantized_model.save('./quantized_yolov4_model.h5')
