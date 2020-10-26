import argparse
import ast
import json

from flask import Flask, request

from translate import load_model_env, translate

app = Flask(__name__)


def validate_args(args):
    # TODO: Validate args.
    pass


def validate_data(data):
    # TODO: Validate data.
    return True


@app.route('/', methods=['POST'])
def serve():
    response = {
        'confidence': None,
        'parse': 'no response'
    }

    if request.method == 'POST':
        data_bytes = request.data
        data_str = data_bytes.decode('UTF-8')
        data = ast.literal_eval(data_str)
        if validate_data(data):
            result = evaluate(data[0])
            response = {
                'confidence': result['confidence'],
                'parse': ''.join(result['parser'].sequence)
            }

    return response


def evaluate(data):
    src = data['src']
    model_id = data['id']
    model_env = models[model_id]['model_env']
    model_opt = models[model_id]['model_opt']
    result = translate(model_env, model_opt, src)
    return result


def run(
    host='localhost',
    port=4996,
    debug=False
):

    app.run(
        host=host,
        port=port,
        debug=debug
    )


def load_models(args):

    models = {}
    with open(args.config) as file:
        config = json.load(file)

    for model_conf in config['models']:
        model_path = model_conf['model']
        model_env = load_model_env(model_path)

        models[model_conf['id']] = {
            'model_env': model_env,
            'model_opt': model_conf['opt']
        }

    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to server config file.')

    parser.add_argument('--host', type=str, default='localhost',
                        help='Path to server config file.')

    parser.add_argument('--port', type=int, default=4996,
                        help='Path to server config file.')

    args = parser.parse_args()
    validate_args(args)

    models = load_models(args)

    run(
        host=args.host,
        port=args.port,
        debug=False
    )
