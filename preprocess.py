import argparse
import torch

from collections import OrderedDict
from datetime import datetime
from lark import Lark, Token, UnexpectedToken

import util.io as io

from model.parser import LALR, filter_unary
from util.log import Logger
from util.nlp import NLP, Vocab, parse_grammar


def validate_args(args):
    """
    Verifies preprocessing.py script arguments.

    :param args:    Script arguments.
    :returns:       'True' if arguments are valid,
                    else 'False'.
    """

    valid = True

    def validate_split_args(src_split, tgt_split):
        # Verify both source and target data paths
        # are set for this split.

        if bool(src_split[1]) ^ bool(tgt_split[1]):
            args_ids = ((tgt_split[0], src_split[0])
                        if src_split[1] is None else
                        (src_split[0], tgt_split[0]))

            logger['log'].log(
                f'[ERR  {datetime.now()}]    ERROR: Invalid arguments, '
                f'\'{args_ids[0]}\' requires \'{args_ids[1]}\''
            )

            return False

        else:
            return True

    src_train = ('src_train', args.src_train)
    tgt_train = ('tgt_train', args.tgt_train)
    valid = validate_split_args(src_train, tgt_train)

    src_dev = ('src_dev', args.src_dev)
    tgt_dev = ('tgt_dev', args.tgt_dev)
    valid = validate_split_args(src_dev, tgt_dev)

    src_test = ('src_test', args.src_test)
    tgt_test = ('tgt_test', args.tgt_test)
    valid = validate_split_args(src_test, tgt_test)

    return valid


def preprocess(args):
    """
    The main preprocessing function. Generates the
    encoder and decoder vocabularies and dataset
    objects. Stores the following files under the
    path and name passed with the 'save_data' argument:

    - <save_data>.<split>.pt
    - <save_data>.lang.pt

    These files serve as input to the 'train.py' script.

    :param args:    Script arguments.
    """

    # Preprocess grammar and build terminal operators.
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

    try:

        # Generate the parser.
        lark = Lark(
            filtered,
            keep_all_tokens=True,
            parser='lalr',
            start=args.start
        )

    except Exception as e:

        logger['log'].log(
            f'[ERR  {datetime.now()}]    ERROR: '
            f'{e.args[0]}. (Wrong start rule argument?)'
        )
        return

    if args.check:
        logger['log'].log(
            f'[INFO {datetime.now()}]    validating datasets'
        )

        nlp = NLP(lark, operators)
        validate_datasets(nlp, datasets)

    else:
        logger['log'].log(
            f'[INFO {datetime.now()}]    building vocabularies'
        )

        nlp = NLP(lark, operators)
        vocab, vocab_dicts = __build_vocab(nlp, datasets)
        nlp.vocab = vocab

        datasets = __preprocess_datasets(nlp, vocab, datasets)
        __save_data(grammar, vocab_dicts, datasets)


def __build_vocab(nlp, datasets):
    """
    Generates the encoder vocabulary (natural language tokens),
    decoder vocabulary (programming language tokens) and stack
    vocabulary (terminal and non-terminal symbols, tokens) by
    parsing each source and target example in each split.

    :param nlp:         nl processing and parsing utils.
    :param datasets:    all dataset splits with source and
                        target samples.
    """

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

    op_i2w, op_w2i = __operator_vocab(nlp, tgt_vocab)

    vocab_dicts.update({'operator': {
        'i2w': op_i2w,
        'w2i': op_w2i
    }})

    op_vocab = Vocab({
        'i2w': op_i2w,
        'w2i': op_w2i
    })

    vocab.update({'stack': stack_vocab})
    vocab.update({'operator': op_vocab})
    return vocab, vocab_dicts


def __preprocess_datasets(nlp, vocab, datasets):
    """
    Extracts a set of fields from each sample pair in each
    dataset split (see 'build_fields').

    :param nlp:         nl processing and parsing utils.
    :param vocab:       encoder, decoder and stack vocabularies.
    :param datasets:    all dataset splits with source and
                        target samples.
    :returns:           preprocessed datasets.
    """

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
            f'[INFO {now}]    {count:<6}/{data_len:>6} '
            f'preprocessing {dataset_name}'
        )

        for sample in samples:

            fields = __build_fields(nlp, sample, vocab)

            count += 1
            logger['line'].update(
                f'[INFO {now}]    {count:<6}/{data_len:>6} '
                f'preprocessing {dataset_name}'
            )

            if fields is None:

                # Fields were not parsable, skip sample.
                continue

            result[dataset_name].append(fields)

        # TODO: Hack, fix logger.
        if dataset_count == len(datasets):
            logger['line'].close()

        else:
            # newline
            logger['log'].log('')

    return result


def validate_datasets(nlp, datasets):
    """
    Checks whether there is an equal number of
    source and target samples in a dataset split.
    Each target sample in each split is parsed to
    verify the sample is syntactically correct.

    :param nlp:         nl processing and parsing utils.
    :param datasets:    all dataset splits with source and
                        target samples.
    """

    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        logger['log'].log(
            f'[INFO {datetime.now()}]    validating dataset '
            f'\'{dataset_name}\''
        )

        sources = dataset['src']
        targets = dataset['tgt']
        success = True

        # Check if equal number of source samples
        # and target samples in dataset.
        if len(sources) != len(targets):
            success = False
            logger['log'].log(
                f'[WARN {datetime.now()}]    sample count mismatch, '
                f'{len(sources)} source samples and {len(targets)} '
                'target samples'
            )

        # Parse each target sample and verify
        # it is a syntactically valid sample.
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
                f' data OK'
            )


def __save_data(grammar, vocab, datasets):
    """
    Saves the following files to under the path and name
    specified as 'save_data' argument.

    - <save_data>.<split>.pt
    - <save_data>.lang.pt

    :param grammar:     the raw grammar file.
    :param vocab:       encoder, decoder, operator and stack
                        vocabularies.
    :param datasets:    all dataset preprocessed dataset splits.
    """

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
    """
    Extracts the set of natural language tokens from
    each source sample in each dataset split and builds
    the encoder vocabulary.

    :param nlp:         nl processing and parsing utils.
    :param datasets:    all dataset splits with source and
                        target samples.
    :returns:           'i2w' and 'w2i' dictionaries taking
                        indices to tokens and vice versa.
    """

    vocab = OrderedDict()

    # Meta-Symbols.
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
    """
    Extracts the set of source code tokens from each target
    sample in each dataset split and builds the decoder
    vocabulary.

    :param nlp:         nl processing and parsing utils.
    :param datasets:    all dataset splits with source and
                        target samples.
    :returns:           'i2w' and 'w2i' dictionaries taking
                        indices to tokens and vice versa.
    """

    vocab = OrderedDict()

    # Meta-Symbols.
    marks = nlp.mark.out.values()
    marks = {repr(m): None for m in marks}
    vocab.update(marks)

    def op_repr(op):
        token = Token(op.name, f'<{op.type}>')
        return repr(token)

    for terminal in nlp.TERMINALS.values():
        # Add all recorded tokens for each terminal.
        tokens = terminal.tokens
        tokens = {repr(t): None for t in tokens}
        vocab.update(tokens)

    for dataset_name in datasets:
        tgt = datasets[dataset_name]['tgt']

        for sample in tgt:
            # Parse each target sample and
            # update vocabulary dict with tokens.

            try:

                tokens = nlp.tokenize(sample)

            except UnexpectedToken:

                # Skip sample if not parsable.
                continue

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
    """
    Collects the set of symbols that can occur on
    the value stack of the parser. Builds the vocabulary
    for the stack encoder.

    :param nlp: nl processing and parsing utils.
    :returns:   'i2w' and 'w2i' dictionaries taking
                indices to tokens and vice versa.
    """

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


def __operator_vocab(nlp, tgt_vocab):
    """
    Collects operators from target vocabulary and stores them
    in a small distinct vocabulary.

    :param nlp:         nl processing and parsing utils.
    :returns:           'i2w' and 'w2i' dictionaries taking
                        indices to tokens and vice versa.
    """

    i2w = {}
    w2i = {}

    for op_name in nlp.OPERATOR:
        op = nlp.OPERATOR[op_name]
        t = repr(op.tokens[0])
        i = tgt_vocab.w2i(t)
        i2w.update({i: t})
        w2i.update({t: i})

    return i2w, w2i


def __build_fields(nlp, sample, vocab):
    """
    Preprocesses each sample in various ways and constructs
    a number of fields used during training.

    :param sample:      current source and target pair to be
                        preprocessed.
    :returns:           dictionary containg generated sample
                        fields.
    """

    src = sample[0]
    tgt = sample[1]

    src_tokens = nlp.normalize(src, delimiters=True)

    try:

        tgt_tokens = nlp.tokenize(tgt)

    except UnexpectedToken:

        # Abort if target sample is not parsable.
        return None

    tgt_tokens = filter_unary(nlp, tgt_tokens)

    # Create a mini sample vocab for copying.
    sample_i2w = {i: t for i, t in enumerate(src_tokens)}
    sample_w2i = {t: i for i, t in enumerate(src_tokens)}
    sample_vocab = {'i2w': sample_i2w, 'w2i': sample_w2i}

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
        'alignment': alignment,
        'value_stacks': value_stacks,
        'stack_lens': stack_lens
    }

    return sample_fields


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--grammar', type=str, required=True,
                        help='Lark grammar for parsing target samples.')

    parser.add_argument('--start', type=str, required=True,
                        help='The start rule of the grammar.')

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
                        help='Check dataset and parse target samples.')

    args = parser.parse_args()

    log = Logger()
    line = log.add_text('')
    log.start()

    logger = {
        'log': log,
        'line': line
    }

    if validate_args(args):
        preprocess(args)
