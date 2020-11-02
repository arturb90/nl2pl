import time

from model.parser import LALR


def elapsed(since):
    '''
    Returns elapsed time since 'since'.
    '''
    now = time.time()
    elapsed = now - since
    return elapsed


def filter_dict(pred, dict_):
    '''
    Filters a dictionary based the predicate provided.
    '''

    for key, val in dict_.items():
        if pred(key, val):
            yield key, val


def parse_incomplete(tokens):
    # TODO: Remove?
    parser = LALR()
    for token in tokens:
        parser.parse(token)
    return parser
