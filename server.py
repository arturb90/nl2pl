'''
This script lets you run multiple models specified
in 'config.json' in the root directory as translation
service.

See 'config.json' for an example.
'''

import argparse
import ast
import json

from flask import Flask, request

from translate import load_model_env, translate

app = None
model_dict = None


def validate_args(args):
    # TODO: Validate args.
    pass


def validate_data(data):
    # TODO: Validate data.
    return True


@app.route('/', methods=['POST'])
def serve():

    # Default response.
    response = {
        'confidence': None,
        'parse': 'no response'
    }

    if request.method == 'POST':

        # Unpack and parse client request data.
        data_bytes = request.data
        data_str = data_bytes.decode('UTF-8')
        data = ast.literal_eval(data_str)

        # Evaluate data if valid.
        if validate_data(data):

            result = evaluate(data[0])
            response = {
                'confidence': result['confidence'],
                'parse': ''.join(result['parser'].sequence)
            }

    return response


def evaluate(model_env, data):
    """
    Evaluate data received from client. Data must
    include a model id and an input source string.

    :param data:    path to config file.
    :returns:       evaluation result.
    """

    src = data['src']
    model_id = data['id']
    model_env = models[model_id]

    # Invoke the translate script.
    result = translate(model_env, src)
    return result


def run(
    models,
    host='localhost',
    port=4996,
    debug=False
):

    global app
    app = Flask(__name__)

    global model_dict
    model_dict = models

    # Start the flask app.
    app.run(
        host=host,
        port=port,
        debug=debug
    )


def load_models(config_path):
    """
    Loads 'config.json' from path and then loads
    models and corresponding model configurations.

    :param config:  path to config file.
    :returns:       loaded models.
    """

    models = {}
    with open(args.config) as file:
        config = json.load(file)

    for model_conf in config['models']:
        model_path = model_conf['model']
        model_env = load_model_env(model_path)
        model_env['model_opt'] = model_conf['opt']

        # Unique ID for each loaded model enables
        # clients to address models individually.
        models[model_conf['id']] = model_env

    return models


def main2(
    models,
    host='localhost',
    port=4996,
    debug=False
):

    run(
        host=args.host,
        port=args.port,
        models=models,
        debug=False
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to server config file.')

    parser.add_argument('--host', type=str, default='localhost',
                        help='Server host address.')

    parser.add_argument('--port', type=int, default=4996,
                        help='Port on which to serve.')

    args = parser.parse_args()
    validate_args(args)

    models = load_models(args.config)

    main(
        host=args.host,
        port=args.port,
        models=models,
        debug=False
    )
