from yolov4_final import make_yolov4_model
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import tensorflow as tf

pretrained_model_path = "./yolov4_save"

model = make_yolov4_model()

# train
mnist_train, info_train = tfds.load('mnist', split='train', shuffle_files=True, with_info=True)
mnist_train = mnist_train.take(1)
images_array = np.empty((0, 608, 608, 3), float)
for example in mnist_train.as_numpy_iterator():
    image = example['image']
    image = np.stack((image)*3, axis=-1)
    image = Image.fromarray(image, 'RGB')
    resized_image = image.resize((608, 608))
    image_data_array = np.asarray(resized_image)
    image_data_array = np.expand_dims(image_data_array, axis=0)
    images_array = np.append(images_array, image_data_array, axis=0)

loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.001 ** 4)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
print(model.summary())
model.fit(x=images_array, y=np.random.randint(0, 2, (1, 1, 80)), epochs=1)
model.save(pretrained_model_path)


