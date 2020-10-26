import argparse
import torch

from collections import OrderedDict
from datetime import datetime
from lark import Lark, Token

import util.io as io

from model.parser import LALR, filter_unary
from util.log import Logger
from util.nlp import NLP, Vocab, parse_grammar


def validate_args(args):
    # TODO: Validate args.
    pass


def preprocess(args):

    grammar = io.grammar(args.grammar)
    filtered, operators = parse_grammar(grammar)

    datasets = io.data(
        src_train=args.src_train,
        tgt_train=args.tgt_train,
        src_dev=args.src_dev,
        tgt_dev=args.tgt_dev,
        src_test=args.src_test,
        tgt_test=args.tgt_test,
    )

    lark = Lark(
        filtered,
        keep_all_tokens=True,
        parser='lalr',
        start=args.start
    )

    if args.check:
        logger['log'].log(
            f'[INFO {datetime.now()}]    validating target data'
        )

        nlp = NLP(lark, operators)
        validate_datasets(nlp, datasets)

    else:
        logger['log'].log(
            f'[INFO {datetime.now()}]    building vocabularies'
        )

        nlp = NLP(lark, operators)
        vocab, vocab_dicts = build_vocab(nlp, datasets)
        nlp.vocab = vocab

        datasets = preprocess_datasets(nlp, vocab, datasets)
        save_data(grammar, vocab_dicts, datasets)


def build_vocab(nlp, datasets):
    inp_i2w, inp_w2i = __input_vocab(nlp, datasets)
    out_i2w, out_w2i = __output_vocab(nlp, datasets)

    vocab_dicts = {
        'src': {
            'i2w': inp_i2w,
            'w2i': inp_w2i,
        },
        'tgt': {
            'i2w': out_i2w,
            'w2i': out_w2i,
        }
    }

    src_vocab = Vocab(vocab_dicts['src'])
    tgt_vocab = Vocab(vocab_dicts['tgt'])

    vocab = {
        'src': src_vocab,
        'tgt': tgt_vocab
    }

    nlp.collect_tokens(vocab)
    stack_i2w, stack_w2i = __stack_vocab(nlp)

    vocab_dicts.update({'stack': {
        'i2w': stack_i2w,
        'w2i': stack_w2i
    }})

    stack_vocab = Vocab({
        'i2w': stack_i2w,
        'w2i': stack_w2i
    })

    vocab.update({'stack': stack_vocab})
    return vocab, vocab_dicts


def preprocess_datasets(nlp, vocab, datasets):
    result = {}
    dataset_count = 0
    for dataset_name in datasets:
        dataset_count += 1
        result[dataset_name] = []
        dataset = datasets[dataset_name]
        src = dataset['src']
        tgt = dataset['tgt']
        samples = zip(src, tgt)
        data_len = len(tgt)
        count = 0

        now = datetime.now()
        logger['line'].update(
            f'[INFO {now}]    {count:<6}/{data_len:>6} ' +
            f'preprocessing {dataset_name}'
        )

        for sample in samples:
            fields = __build_fields(nlp, sample, vocab)
            result[dataset_name].append(fields)
            count += 1

            logger['line'].update(
                f'[INFO {now}]    {count:<6}/{data_len:>6} ' +
                f'preprocessing {dataset_name}'
            )

        # TODO: Hack, fix logger.
        if dataset_count == len(datasets):
            logger['line'].close()

        else:
            # newline
            logger['log'].log('')

    return result


def validate_datasets(nlp, datasets):

    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        logger['log'].log(
            f'[INFO {datetime.now()}]    validating targets in '
            f'\'{dataset_name}\''
        )

        if 'tgt' in dataset:
            targets = dataset['tgt']
            success = True

            for i in range(len(targets)):

                try:
                    nlp.lark.parse(targets[i])

                except Exception:
                    success = False
                    logger['log'].log(
                        f'[WARN {datetime.now()}]    parsing error '
                        f'while parsing line {i+1}'
                    )

            if success:
                logger['log'].log(
                    f'[INFO {datetime.now()}]    \'{dataset_name}\' '
                    f'target data OK'
                )


def save_data(grammar, vocab, datasets):
    lang = {
        'grammar': grammar,
        'start': args.start,
        'vocab': vocab
    }

    lang_path = f'{args.save_data}.lang.pt'
    torch.save(lang, lang_path)

    logger['log'].log(
        f'[INFO {datetime.now()}]    vocab stored in '
        f'\'{lang_path}\''
    )

    for k, v in datasets.items():
        data_path = f'{args.save_data}.{k}.pt'
        torch.save(v, data_path)

        logger['log'].log(
            f'[INFO {datetime.now()}]    {k} dataset '
            f'stored in \'{data_path}\''
        )


def __input_vocab(nlp, datasets):
    vocab = OrderedDict()

    marks = nlp.mark.inp.values()
    marks = {m: None for m in marks}
    vocab.update(marks)

    for dataset_name in datasets:
        src = datasets[dataset_name]['src']

        for sample in src:
            tokens = nlp.normalize(sample)
            tokens = {t: None for t in tokens}
            vocab.update(tokens)

    vocab = [t for t in vocab]
    i2w = {i: t for i, t in enumerate(vocab)}
    w2i = {t: i for i, t in enumerate(vocab)}
    return i2w, w2i


def __output_vocab(nlp, datasets):
    vocab = OrderedDict()

    marks = nlp.mark.out.values()
    marks = {repr(m): None for m in marks}
    vocab.update(marks)

    def op_repr(op):
        token = Token(op.name, f'<{op.type}>')
        return repr(token)

    for terminal in nlp.TERMINALS.values():
        tokens = terminal.tokens
        tokens = {repr(t): None for t in tokens}
        vocab.update(tokens)

    for dataset_name in datasets:
        tgt = datasets[dataset_name]['tgt']

        for sample in tgt:
            tokens = nlp.tokenize(sample)
            tokens = {
                (repr(t) if t.type not in nlp.OPERATOR
                 else op_repr(nlp.OPERATOR[t.type])): None
                for t in tokens
            }

            vocab.update(tokens)

    vocab = [t for t in vocab]
    i2w = {i: t for i, t in enumerate(vocab)}
    w2i = {t: i for i, t in enumerate(vocab)}
    return i2w, w2i


def __stack_vocab(nlp):
    vocab = OrderedDict()

    marks = nlp.mark.out.values()
    marks = {repr(m): None for m in marks}
    vocab.update(marks)

    nonterminals = nlp.NONTERMINALS.values()
    symbols = {repr(nt.nt): None for nt in nonterminals}
    vocab.update(symbols)

    terminals = nlp.TERMINALS.values()
    symbols = {repr(t): None for t in terminals}
    vocab.update(symbols)

    tokens = nlp.TOKENS.values()
    tokens = {repr(t): None for t in tokens}
    vocab.update(tokens)

    i2w = {i: t for i, t in enumerate(vocab)}
    w2i = {t: i for i, t in enumerate(vocab)}

    return i2w, w2i


def __build_fields(nlp, sample, vocab):
    src = sample[0]
    tgt = sample[1]

    src_tokens = nlp.normalize(src, delimiters=True)
    tgt_tokens = nlp.tokenize(tgt, delimiters=True)
    tgt_tokens = filter_unary(nlp, tgt_tokens)

    # Create a mini sample vocab for copying. The target
    # vocabulary is dynamically extended by this sample
    # vocab in case copy attention is used.
    sample_i2w = {i: t for i, t in enumerate(src_tokens)}
    sample_w2i = {t: i for i, t in enumerate(src_tokens)}
    sample_vocab = {'i2w': sample_i2w, 'w2i': sample_w2i}
    tgt_vocab_ext = vocab['tgt'].extend(sample_vocab, copy=True)

    # Create alignment vector specifying which target
    # tokens should be copied from the input sequence
    # and replace operator tokens in target sequence.
    alignment = nlp.alignment(src_tokens, tgt_tokens, sample_vocab)

    # Replace target tokens of operator type with
    # respective placeholder.
    for i in range(len(tgt_tokens)):
        if tgt_tokens[i].type in nlp.OPERATOR:
            op = nlp.OPERATOR[tgt_tokens[i].type]
            tgt_tokens[i] = Token(op.name, f'<{op.type}>')

    # Nonterminals in the value stack for each decoding step.
    parser = LALR(nlp)
    stack_seq = nlp.stack_sequence(parser.value_stack, filter_token=True)
    stack_i = nlp.stack2indices(stack_seq, delimiters=True)
    value_stacks = [stack_i]

    for token in tgt_tokens[1:]:
        parser.parse(token)
        stack_seq = nlp.stack_sequence(parser.value_stack, filter_token=True)
        stack_i = nlp.stack2indices(stack_seq, delimiters=True)
        value_stacks.append(stack_i)

    # Pre-pad value stacks.
    max_stack_len = max(len(vs) for vs in value_stacks)
    stack_lens = [len(vs) for vs in value_stacks]
    for i in range(len(value_stacks)):
        out = [0] * max_stack_len
        out[:len(value_stacks[i])] = value_stacks[i]
        value_stacks[i] = out

    src_i = [vocab['src'].w2i(t) for t in src_tokens]
    tgt_i = [vocab['tgt'].w2i(repr(t)) for t in tgt_tokens]

    sample_fields = {
        'src': src,
        'tgt': tgt,
        'src_i': src_i,
        'tgt_i': tgt_i,
        'sample_vocab': sample_vocab,
        'tgt_vocab_ext': tgt_vocab_ext,
        'alignment': alignment,
        'value_stacks': value_stacks,
        'stack_lens': stack_lens
    }

    return sample_fields


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--grammar', type=str, required=True,
                        help='Lark grammar for parsing target samples.')

    parser.add_argument('--start', type=str, default='start',
                        help='The start rule of the grammar. Defaults to "start".')

    parser.add_argument('--src_train', type=str, default=None,
                        help='Training dataset source samples.')

    parser.add_argument('--tgt_train', type=str, default=None,
                        help='Training dataset target samples.')

    parser.add_argument('--src_test', type=str, default=None,
                        help='Test dataset source samples.')

    parser.add_argument('--tgt_test', type=str, default=None,
                        help='Test dataset target samples.')

    parser.add_argument('--src_dev', type=str, default=None,
                        help='Development dataset source samples.')

    parser.add_argument('--tgt_dev', type=str, default=None,
                        help='Development dataset target samples.')

    parser.add_argument('--save_data', type=str, required=True,
                        help='Path and name for saving preprocessed data.')

    parser.add_argument('--check', action='store_true', default=False,
                        help='Check whether target examples are valid programs.')

    args = parser.parse_args([
        '--grammar',        'data/grammars/geoquery-sql-vars_kept.lark',
        '--src_train',      'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-src_train.txt',
        '--tgt_train',      'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-tgt_train.txt',
        '--src_dev',        'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-src_dev.txt',
        '--tgt_dev',        'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-tgt_dev.txt',
        '--src_test',       'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-src_test.txt',
        '--tgt_test',       'data/datasets/geoquery/geo_sql/question_split/vars_kept/geo_sql-a-tgt_test.txt',
        '--save_data',      'compiled/geoquery_sql'
    ])

    # args = parser.parse_args([
    #     '--grammar',        'data/grammars/atis_sql.lark',
    #     '--src_train',      'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-src_train.txt',
    #     '--tgt_train',      'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-tgt_train.txt',
    #     '--src_dev',        'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-src_dev.txt',
    #     '--tgt_dev',        'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-tgt_dev.txt',
    #     '--src_test',       'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-src_test.txt',
    #     '--tgt_test',       'data/datasets/atis/atis_sql/question_split/vars_replaced/atis_sql-tgt_test.txt',
    #     '--save_data',      'compiled/atis/atis_sql',
    #     '--check'
    # ])

    log = Logger()
    line = log.add_text('')
    log.start()

    logger = {
        'log': log,
        'line': line
    }

    validate_args(args)
    preprocess(args)
