import time

from model.parser import LALR


def elapsed(since):
    now = time.time()
    elapsed = now - since
    return elapsed


def filter_dict(pred, dict_):
    for key, val in dict_.items():
        if pred(key, val):
            yield key, val


def parse_incomplete(tokens):
    parser = LALR()
    for token in tokens:
        parser.parse(token)
    return parser
