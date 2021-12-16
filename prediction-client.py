#!?usr/bin/env python3

import tensorflow as tf
import numpy as np
import requests
import json


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


data = {
    'signature_name': 'serving_default',
    'instances': {
        'image_data': test_images[:5].tolist()
    }
}

predictions = requests.post(
    'http://127.0.0.1:8501/v1/models/fashion-mnist:predict',
    json=data
)
predictions = json.loads(predictions.text)[0]

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(f'Predicted class: {class_names[np.argmax(predictions[0])]}')
print(f'Actual class: {class_names[test_labels[0]]}')
