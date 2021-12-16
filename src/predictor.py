#!/usr/bin/env python3

import os
import json
import flask
import importlib.util
import logging
import traceback
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.WARN)


class RobustEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)


def not_found_on_error(handler):
    def new_handler(*args, **kwargs):
        try:
            res, status = handler(*args, **kwargs)
        except:
            e_repr = traceback.format_exc()
            logger.error(e_repr)
            res = {
                'state': 'UNAVAILABLE',
                'status': {'error_code': 'UNKNOWN', 'error_message': e_repr},
            }
            status = 404
        return flask.Response(
            response=json.dumps(res, cls=RobustEncoder),
            status=status,
            mimetype='application/json'
        )
    new_handler.__name__ = handler.__name__
    return new_handler


class ScoringService(object):
    models = {}

    @classmethod
    def get_model(cls, model: str):
        def get_fs_models():
            return os.listdir('/opt/models/')

        if model not in cls.models:
            assert model in get_fs_models(), f'model not found: {model}'
            spec = importlib.util.spec_from_file_location(
                f'{model}',
                f'/opt/models/{model}/{model}.py'
            )

            cls.models[model] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.models[model])

        return cls.models[model]

    @classmethod
    def predict(cls, model, input):
        clf = cls.get_model(model)
        return clf.Model.predict(input)

    @classmethod
    def metadata(cls, model):
        clf = cls.get_model(model)
        return clf.Model.metadata()


app = flask.Flask(__name__)


@app.route('/v1/models/<model>', methods=['GET'])
@not_found_on_error
def ping(model):
    model = ScoringService.get_model(model)
    status = 200
    res = {
        'model_status': {
            'state': 'AVAILABLE',
            'status': {'error_code': 'OK', 'error_message': ''},
        }
    }
    return res, status


@app.route(
    '/v1/models/<model>/metadata',
    methods=['GET']
)
@not_found_on_error
def metadata(model):
    metadata = ScoringService.metadata(model)
    assert metadata is not None, f'model {model} returned empty metadata'
    return metadata, 200


@app.route(
    '/v1/models/<model>:predict',
    methods=['POST']
)
@not_found_on_error
def predict(model):
    body = flask.request.json

    model_spec = ScoringService.metadata(model)
    schema = model_spec['inputs']
    assert ('instances' in body)
    parsed_data = {}
    data = body['instances']
    record_count = len(data)

    extraneous_fields = set(data.keys()) - set(schema.keys())
    assert not extraneous_fields, (
        f'received extraneous fields {extraneous_fields}')

    for field, info in schema.items():
        parsed_data[field] = np.asarray(data[field], dtype=info['dtype'])

    # Do the prediction
    return ScoringService.predict(model, parsed_data), 200


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return flask.Response(
        response='Model server is running!',
        status=200,
        mimetype='text/html'
    )
