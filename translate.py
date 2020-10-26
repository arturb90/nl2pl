import argparse
import torch

from datetime import datetime
from lark import Lark

from torch.nn.utils.rnn import pad_sequence

from model import build_model, model_settings
from model.statistics import Scorer
from util.log import Logger
from util.nlp import NLP, Vocab, parse_grammar

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def validate_args(args):
    # TODO: Validate args.
    pass


def load_model_env(model_path):
    model_data = torch.load(
        model_path,
        map_location=device
    )

    model_opt = model_data['model_opts']
    lang = model_data['lang']
    vocab = {
        'src': Vocab(lang['vocab']['src']),
        'tgt': Vocab(lang['vocab']['tgt']),
        'stack': Vocab(lang['vocab']['stack'])
    }

    grammar, operators = parse_grammar(lang['grammar'])

    lark = Lark(
        grammar,
        keep_all_tokens=True,
        parser='lalr',
        start='start'
    )

    nlp = NLP(lark, operators)
    nlp.collect_tokens(vocab)
    nlp.vocab = vocab

    settings = model_settings(vocab, model_opt)
    model = build_model(vocab, settings)
    model.load_state_dict(model_data['state_dict'])
    model.to(model.device)
    model.eval()

    model_env = {
        'model': model,
        'vocab': vocab,
        'nlp': nlp
    }

    return model_env


def translate(model_env, model_opt, src):
    model = model_env['model']
    vocab = model_env['vocab']
    nlp = model_env['nlp']

    src_i = []
    tokens = nlp.normalize(src, delimiters=True)
    unk_token = nlp.mark.inp['UNK']
    unk_index = vocab['src'].w2i(unk_token)
    opts = get_opts(model_opt)

    for t in tokens:

        try:
            i = vocab['src'].w2i(t)
            src_i.append(i)

        except KeyError:
            src_i.append(unk_index)

    results = model.evaluate(
        nlp, torch.LongTensor(src_i).to(model.device),
        tokens, num_parsers=opts['beam_width'],
        beam_width=opts['beam_width']
    )

    return results


def get_opts(model_opt):
    # Default options.
    options = {
        'beam_width': 1
    }

    if 'beam_width' in model_opt:
        options['beam_width'] = model_opt['beam_width']

    return options


def evaluate(args, env, dataset, logger):
    model = env['model']
    vocab = env['vocab']
    nlp = env['nlp']

    logger['log'].log(
        f'[INFO {datetime.now()}]    evaluating model '
        f'\'{args.model}\' on dataset \'{args.eval}\''
    )

    count = 0
    data_len = len(dataset)

    now = datetime.now()
    logger['line'].update(
        f'[INFO {now}]    {count:<4}/{data_len:>4} '
        f'examples evaluated'
    )

    scorer = Scorer(nlp, vocab)
    for example in dataset:
        tgt = example['tgt_i'][1:]

        if args.no_parser:
            # Evaluate model without parser assistance.
            # TODO: Gives wrong results when stack
            # encodings were used.
            alignment = example['alignment']
            tgt_i = torch.LongTensor(example['tgt_i'])
            src_i = torch.LongTensor(example['src_i'])
            tgt_i = tgt_i.unsqueeze(1).to(model.device)
            src_i = src_i.unsqueeze(1).to(model.device)

            stacks = torch.LongTensor(example['value_stacks']).to(model.device)
            stack_lens = torch.LongTensor(example['stack_lens'])
            stacks = stacks.unsqueeze(0)

            stack_lens = pad_sequence((stack_lens,), padding_value=1)
            stack_lens = stack_lens.tolist()

            # stacks = example['value_stacks']
            # stack_lens = [[len(vs)] for vs in stacks]
            # stacks = torch.LongTensor(stacks).to(model.device)
            # stacks = stacks.unsqueeze(0)

            with torch.no_grad():
                output = model(
                    src_i, tgt_i,
                    [len(src_i)], [len(tgt_i)],
                    alignment, stacks,
                    stack_lens, 0.0
                )

            dec_outs = output['dec_outs']
            vocab_size = model.decoder.vocab_size
            dec_outs = dec_outs[1:, :, :]
            dec_outs = dec_outs.squeeze()
            predictions = dec_outs.argmax(1)

        else:
            # Evaluate model with parser assistance.
            src_i = torch.LongTensor(example['src_i'])
            src_i.to(model.device)

            top = model.evaluate(
                nlp, src_i.to(model.device), example['src'],
                num_parsers=args.beam_width,
                beam_width=args.beam_width
            )

            predictions = top['parser'].predictions

        results = {
            'predictions': predictions,
            'attn_used': False
        }

        if model.attention:
            results.update({
                'attn_used': True
            })

        scorer.score(results, tgt)
        count += 1

        logger['line'].update(
            f'[INFO {now}]    {count:<4}/{data_len:>4} '
            f'examples evaluated'
        )

    logger['line'].close()
    return scorer.results()


def main(args, logger):
    env = load_model_env(args.model)

    if args.eval:
        dataset = torch.load(args.eval)
        scores = evaluate(args, env, dataset, logger)

        logger['log'].log(
            f'[INFO {datetime.now()}]    {scores["accuracy"]*100:0>6.3f}% '
            f'accuracy on dataset \'{args.eval}\''
        )

        logger['log'].log(
            f'[INFO {datetime.now()}]    {scores["gold_acc"]*100:0>6.3f}% '
            f'gold accuracy on dataset \'{args.eval}\''
        )

        logger['log'].close()

    else:
        # TODO: Run translation server.
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        help='The model file to use for translation.')

    parser.add_argument('--eval', type=str, default=None,
                        help='The test dataset to evaluate.')

    parser.add_argument('--out', type=str, default=None,
                        help='The logging file.')

    parser.add_argument('--beam_width', type=int, default=1,
                        help='The beam with for the parser decoder. Defaults \
                            to greedy search.')

    parser.add_argument('--no_parser', action='store_true',
                        help='Turns off parser-assisted decoding.')

    args = parser.parse_args([
        '--model',          'compiled/geoquery-model.model_step_403.pt',
        '--eval',           'compiled/geoquery_sql.test.pt',
        '--out',            'compiled/log_eval.txt',
        '--beam_width',     '1',
        '--no_parser'
    ])

    # args = parser.parse_args([
    #     '--model',          '../nl2pl-demo/model/geoquery_demo.model.pt',
    #     '--beam_width',     '1'
    # ])

    log = Logger(out_path=args.out)
    line = log.add_text('')
    log.start()

    logger = {
        'log': log,
        'line': line
    }

    validate_args(args)
    main(args, logger)
