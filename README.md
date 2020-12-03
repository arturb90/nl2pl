# nl2pl

[![Licence](https://img.shields.io/badge/license-MIT-orange)](https://gitlab.dlr.de/bara_at/nl2pl/-/blob/master/LICENSE)

nl2pl is a tool for training neural semantic parsers that predict program code from natural language descriptions. It uses a grammar specification and a LALR(1) parser to enforce syntactically valid programs during inference. Check out the [live demo](https://safe-plateau-06076.herokuapp.com/) (hosted on a Heroku free dyno, may take a few seconds to boot) to test some example models.


## Requirements

As of now, Python3 is supported. nl2pl depends on [lark](https://github.com/lark-parser/lark) for parsing, [pytorch](https://github.com/pytorch/pytorch) for building neural network models. Set up a new virtual environment and install all requirements:

```
pip install -r requirements.txt
```


## Quickstart

The tool employs a parser-guided decoder for predicting tokens. Accordingly, each model you build will be language-specific. The parser is generated using [lark](https://github.com/lark-parser/lark). In order to preprocess your datasets, you have to provide a `.lark` file containing your grammar specification. The [lark grammar reference](https://lark-parser.readthedocs.io/en/latest/grammar/) contains detailed documentation in this regard. 

#### 1) Preprocessing the data

```
python3 preprocess.py \
    --grammar your_grammar.lark \
    --src_train path/to/src_train.txt \
    --tgt_train path/to/tgt_train.txt \
    --src_dev path/to/src_dev.txt \
    --tgt_dev path/to/tgt_dev.txt \
    --src_test path/to/src_test.txt \
    --tgt_test path/to/tgt_test.txt \
    --save_data data_name
```

The preprocessing script expects at least a grammar, the dataset files containing training sources (natural language descriptions) and targets (corresponding program code) and a name under which to store the preprocessed data. Optionally, development and test splits can be provided for validating training runs and inference. The dataset source and target files are expected to be parallel, containing one example per line. The script will yield the following files:

* `data_name.train.pt`
* `data_name.dev.pt`
* `data_name.test.pt`
* `data_name.lang.pt`

Note that during preprocessing, target examples that cannot be parsed according to the provided grammar will be discarded. You can validate your target examples and spot faulty programs in your dataset:

```
python3 preprocess.py \
    --grammar your_grammar.lark \
    --tgt_train path/to/tgt_train.txt \
    --tgt_dev path/to/tgt_dev.txt \
    --tgt_test path/to/tgt_test.txt \
    --check
```

#### 2) Training the model

```
python3 train.py --data /path/to/data/data_name --save model_name --validate
```

The training script defaults to a simple sequence-to-sequence model with an unidirectional one-layer encoder and decoder. By adding the `--attention` flag [Bahdanau attention](https://arxiv.org/abs/1409.0473) will be used for decoding. By adding the `--copy` flag a [pointer-generator network](https://arxiv.org/abs/1704.04368) will be used for copying tokens from the input sentence. See `train.py` for a list of all possible configuration options. If the `--validate` flag is set, the script will validate training results on the development data and save the model with best performance on the development data.

The script will yield a model ready for inference:

* `model_name.model.pt`

#### 3) Evaluating the model

```
python3 translate.py --model model_name.model.pt --eval path/to/data/data_name.test.pt --out out_file.txt
```

This will evaluate the model on the test split of the dataset and print the statistics to `out_file.txt`. 

#### 4) Running a translation server for inference

```
python3 translate.py --model model_name.model.pt --host <host_ip> --port <port>
```

This will load your model as a translation service. It expects data in the following format:

```
[{"src": "your input sentence to translate", "id": 0}]
```

Test the translation service by sending a POST request:

```
curl -i -X POST -H "Content-Type: application/json"  \
    -d '[{"src": "your input sentence to translate", "id": 0}]' \
    http://<host_ip>:<port>/
```

You can run multiple models by setting a configuration in `config.json` and running `server.py`. See `config.json` for an example.

## Acknowledgements

* The parser implementation is based on [lark](https://github.com/lark-parser/lark)
* The tool design is loosely based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
