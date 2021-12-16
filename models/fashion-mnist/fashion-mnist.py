import onnxruntime as rt
import os
import numpy as np
from pathlib import Path


ROOT = Path(os.path.realpath(os.path.expanduser(__file__))).parents[0]
SESS = rt.InferenceSession(str(ROOT / "fashion-mnist.onnx"))


class Model(object):
    onnx_session = SESS
    input_name = onnx_session.get_inputs()[0].name

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

    @staticmethod
    def metadata():
        return {
            'signature_name': 'serving_default',
            'inputs': {
                'image_data': {
                    'dtype': 'float'
                }
            },
            'outputs': {
                'class_probabilities': {
                    'dtype': 'float'
                }
            }
        }

    @classmethod
    def predict(cls, data):
        y_pred = cls.onnx_session.run(
            None,
            {
                cls.input_name: data['image_data'].astype(np.float32)
            }
        )

        return y_pred
