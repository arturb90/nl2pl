import os
import argparse
import time
import torch

from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader

import util.io as io

from model import build_model, model_settings
from model.statistics import Statistics
from util.misc import elapsed
from util.nlp import Vocab
from util.data import Dataset, collate_fn
from util.log import Logger

# For padded copy weights to ensure
# numerical stability.
_EPSILON = 1e-6


def validate_args(args):
    # TODO: Validate args.
    return True


def __save_model(model, model_opts, lang, epoch, best_epoch):
    """
    Saves the model with the complete data regenerate the
    environment (parser, utilites.). Yields a .pt file.

    :param model_opts:  model configuration data.
    :param lang:        language data such as vocabularies and
                        grammar for parser generation.
    :param epoch:       current training epoch.
    :param best_epoch:  training epoch of the last model saved
                        for removal.
    """

    model_path = f'{args.save}.model_step_{epoch}.pt'
    previous_best = f'{args.save}.model_step_{best_epoch}.pt'

    if os.path.exists(previous_best):
        os.remove(previous_best)

    model_data = {
        'state_dict': model.state_dict(),
        'model_opts': model_opts,
        'lang': lang
    }

    torch.save(model_data, model_path)


def validate(env, dataset, crit):
    """
    Validates the latest model on dataset.

    :param env:         environment including model
                        and language data.
    :param dataset:     the validation dataset.
    :param epoch:       NLL loss criterion.
    :returns:           validation statistics.
    """

    model = env['model']
    lang = env['lang']

    # Set evaluation mode to turn
    # dropout off.
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    dev_loss = 0
    dataiter = iter(dataloader)
    results = []

    # No gradient computation during validation.
    with torch.no_grad():

        for batch in dataiter:
            src_pad, tgt_pad, src_lens, tgt_lens, \
                align_pad, stack_pad, stack_lens = batch

            if model.decoder.stack_encoder:
                # When stack encodings are used,
                # teacher forcing required.
                tf = 1.0

            else:
                # No teacher forcing during validation.
                tf = 0.0

            tgt_len = tgt_pad.size(0)
            src_len = src_pad.size(0)

            output = model(
                src_pad, tgt_pad,
                src_lens, tgt_lens,
                align_pad, stack_pad,
                stack_lens, tf
            )

            dec_outs = output['dec_outs']
            vocab_size = model.decoder.vocab_size
            preds = dec_outs[1:].transpose(0, 1)
            preds = preds.reshape(-1, vocab_size)
            targets = tgt_pad[1:].transpose(0, 1)
            targets = targets.reshape(-1)

            batch_loss = crit(preds, targets)

            if model.copy_attention:
                copy_weights = output['copy_weights']
                copy_weights = copy_weights[1:].transpose(0, 1)
                copy_pred = copy_weights.reshape(-1, src_len)
                # Copy weights are padded and masked, setting
                # padding positions to a very low value ensures
                # numerical stability.
                copy_pred[copy_pred == 0] = _EPSILON
                # Copy weights are softmaxed. But NLL loss
                # expects log-likelihoods.
                copy_pred = torch.log(copy_pred)
                align_pad = align_pad[1:].transpose(0, 1)
                copy_tgts = align_pad.reshape(-1)
                copy_loss = crit(copy_pred, copy_tgts)
                # Add copy loss to batch loss.
                batch_loss += copy_loss

            batch_results = {
                'copy_attn_used': False,
                'tgt_len': tgt_len-1,
                'tgt_vocab_size': vocab_size,
                'predictions': preds,
                'targets': targets
            }

            if model.copy_attention:
                batch_results.update({
                    'copy_attn_used': True,
                    'copy_predictions': copy_pred,
                    'copy_targets': copy_tgts
                })

            # Cache batch results for computing
            # statistics later.
            results.append(batch_results)
            dev_loss += batch_loss.item()

        statistics = Statistics(
            lang, dev_loss,
            len(dataiter),
            results
        )

    return statistics


def train_epoch(env, dataset, opt, crit, epoch_n):
    """
    Training for one epoch.

    :param env:         environment including model
                        and language data.
    :param dataset:     training dataset
    :param opt:         SGD optimizer.
    :param crit:        NLL loss criterion.
    :param epoch_n:     epoch number.
    :returns:           training statistics.
    """

    model = env['model']
    lang = env['lang']

    # Set to training mode.
    model.train()

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    epoch_loss = 0
    dataiter = iter(dataloader)
    dataiter_len = len(dataiter)
    results = []
    count = 0

    now = datetime.now()
    logger['line'].update(
        f'[INFO {now}]    EPOCH {epoch_n} >   '
        f'{count:<4}/{dataiter_len:>4} batches processed'
    )

    for batch in dataiter:
        opt.zero_grad()
        src_pad, tgt_pad, src_lens, tgt_lens, \
            align_pad, stack_pad, stack_lens = batch

        tf = args.teacher_forcing
        tgt_len = tgt_pad.size(0)
        src_len = src_pad.size(0)

        output = model(
            src_pad, tgt_pad,
            src_lens, tgt_lens,
            align_pad, stack_pad,
            stack_lens, tf
        )

        dec_outs = output['dec_outs']
        vocab_size = model.decoder.vocab_size
        preds = dec_outs[1:].transpose(0, 1)
        preds = preds.reshape(-1, vocab_size)
        targets = tgt_pad[1:].transpose(0, 1)
        targets = targets.reshape(-1)

        batch_loss = crit(preds, targets)

        if model.copy_attention:
            copy_weights = output['copy_weights']
            copy_weights = copy_weights[1:].transpose(0, 1)
            copy_pred = copy_weights.reshape(-1, src_len)
            # Copy weights are padded and masked, setting
            # padding positions to a very low value ensures
            # numerical stability.
            copy_pred[copy_pred == 0] = _EPSILON
            # Copy weights are softmaxed. But NLL loss
            # expects log-likelihoods.
            copy_pred = torch.log(copy_pred)
            align_pad = align_pad[1:].transpose(0, 1)
            copy_tgts = align_pad.reshape(-1)
            copy_loss = crit(copy_pred, copy_tgts)
            # Add copy loss to batch loss.
            batch_loss += copy_loss

        batch_loss.backward()

        # Gradient clip to avoid
        # exploding gradients.
        nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip
        )

        opt.step()

        batch_results = {
            'copy_attn_used': False,
            'tgt_len': tgt_len-1,
            'tgt_vocab_size': vocab_size,
            'predictions': preds,
            'targets': targets
        }

        if model.copy_attention:
            batch_results.update({
                'copy_attn_used': True,
                'copy_predictions': copy_pred,
                'copy_targets': copy_tgts
            })

        results.append(batch_results)
        epoch_loss += batch_loss.item()

        if epoch_loss < 0:
            print('pause')

        count += 1
        logger['line'].update(
            f'[INFO {now}]    EPOCH {epoch_n} >   '
            f'{count:<4}/{dataiter_len:>4} batches processed'
        )

    if epoch_n == args.epochs:
        logger['line'].close()

    else:
        # newline
        logger['log'].log('')

    statistics = Statistics(
        lang, epoch_loss,
        dataiter_len,
        results
    )

    return statistics


def train(env, datasets):
    """
    Trains a semantic parser that translates natural
    language expressions to program code based on the
    language data provided.

    :param env:         environment including model and
                        language data.
    :param datasets:    training and validation datasets.
    """

    model = env['model']
    lang = env['lang']

    # Zero is padding token and no alignment.
    crit = nn.NLLLoss(ignore_index=0, reduction='sum')
    opt = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9
    )

    train_data = datasets['train']
    train_set = Dataset(
        train_data,
        model.device,
        args.mask_ratio
    )

    if 'dev' in datasets:
        dev_data = datasets['dev']
        dev_set = Dataset(
            dev_data,
            model.device,
            args.mask_ratio
        )

        best_dev_acc = 0
        best_epoch = 0

    logger['log'].log(
        f'[INFO {datetime.now()}]    commencing '
        f'training for {args.epochs} epochs'
    )

    # space
    print('')
    early_stop = 0
    for epoch in range(1, args.epochs+1):
        since = time.time()
        statistics = train_epoch(
            env, train_set,
            opt, crit, epoch
        )

        duration = elapsed(since)
        loss = statistics.loss
        accuracy = statistics.accuracy
        gold_acc = statistics.gold_accuracy

        logger['log'].log(
            f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
            f'{"elapsed time: ":<25}{duration:.3f}s'
        )

        logger['log'].log(
            f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
            f'{"train loss: ":<25}{loss:.5f}'
        )

        logger['log'].log(
            f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
            f'{"train accuracy: ":<25}{accuracy*100:0>6.3f}%'
        )

        logger['log'].log(
            f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
            f'{"train gold acc.: ":<25}{gold_acc*100:0>6.3f}%'
        )

        if 'dev' in datasets and args.validate:
            # Validate model.
            statistics = validate(env, dev_set, crit)

            dev_loss = statistics.loss
            accuracy = statistics.accuracy
            gold_acc = statistics.gold_accuracy

            logger['log'].log(
                f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                f'{"dev loss: ":<25}{dev_loss:.5f}'
            )

            logger['log'].log(
                f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                f'{"dev accuracy: ":<25}{accuracy*100:0>6.3f}%'
            )

            logger['log'].log(
                f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                f'{"dev gold acc.: ":<25}{gold_acc*100:0>6.3f}%'
            )

            logger['log'].log(
                f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                f'{"best dev accuracy: ":<25}{best_dev_acc*100:0>6.3f}%'
            )

            # Save model if new best exact match accuracy on
            # development set.
            if args.best_gold and gold_acc > best_dev_acc:
                best_dev_acc = gold_acc
                __save_model(model, args, lang, epoch, best_epoch)
                best_epoch = epoch
                early_stop = 0

                logger['log'].log(
                    f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                    f'new best dev split gold accuracy, saving model'
                )

            # Save model if new best accuracy on development set.
            elif not args.best_gold and accuracy > best_dev_acc:
                best_dev_acc = accuracy
                __save_model(model, args, lang, epoch, best_epoch)
                best_epoch = epoch
                early_stop = 0

                logger['log'].log(
                    f'[INFO {datetime.now()}]    EPOCH {epoch} >   '
                    f'new best dev split accuracy, saving model'
                )

            else:
                early_stop = early_stop + 1

        # space
        print('')

        if early_stop == args.early_stop:
            logger['log'].log(
                f'[INFO {datetime.now()}]    no dev set improvement '
                f'since {args.early_stop} epochs, stop training'
            )
            break

    logger['log'].log(
        f'[INFO {datetime.now()}]    training concluded'
    )

    logger['log'].close()

    # Save model each epoch if not validating.
    if 'dev' not in datasets or not args.validate:
        __save_model(model, args, lang, epoch, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General training data and configuration.
    parser.add_argument('--data', type=str, required=True,
                        help='The input datasets and vocabularies.')

    parser.add_argument('--save', type=str, required=True,
                        help='The name under which the model will be saved.')

    parser.add_argument('--out', type=str, default=None,
                        help='The file in which training info is logged.')

    parser.add_argument('--validate', action='store_true', default=False,
                        help='Whether to validate training progress on the'
                        ' dev split')

    parser.add_argument('--early_stop', type=int, default=100,
                        help='Stop training when validation accuracy has not'
                        ' improved since the specified number of iterations.')

    parser.add_argument('--best_gold', action='store_true', default=False,
                        help='Save model with best development gold accuracy'
                        ' when validating.')

    # Global training hyperparameters.
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training iterations.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of samples to batch.')

    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for SGD optimizer.')

    parser.add_argument('--gradient_clip', type=float, default=2,
                        help='Clipping to prevent exploding gradients.')

    parser.add_argument('--mask_ratio', type=float, default=0.0,
                        help='Ratio of input sample tokens to be masked'
                        ' randomly as <UNK> tokens.')

    # Settings for model architecture.
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Attention mechanism according to Bahdanau.')

    parser.add_argument('--copy', action='store_true', default=False,
                        help='Copy attention for copying tokens from the'
                        ' input sentence.')

    # Parameters for value stack encoder.
    parser.add_argument('--stack_encoding', action='store_true', default=False,
                        help='Value stack encodings used during decoding.')

    parser.add_argument('--stack_emb_size', type=int, default=16,
                        help='Dimension of embedding vector for stack'
                        ' encoder.')

    parser.add_argument('--stack_hidden_size', type=int, default=16,
                        help='Dimension of stack encoder hidden state.')

    parser.add_argument('--stack_dropout', type=float, default=0.1,
                        help='Dropout applied to stack encoder embeddings.')

    # Module specific hyperparameters.
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of layers to use for encoder and decoder')

    parser.add_argument('--enc_emb_size', type=int, default=64,
                        help='Dimension of embedding vector for encoder.')

    parser.add_argument('--dec_emb_size', type=int, default=64,
                        help='Dimension of embedding vector for decoder.')

    parser.add_argument('--enc_hidden_size', type=int, default=92,
                        help='Dimension of encoder hidden state.')

    parser.add_argument('--dec_hidden_size', type=int, default=92,
                        help='Dimension of decoder hidden state.')

    parser.add_argument('--enc_emb_dropout', type=float, default=0.1,
                        help='Dropout applied to encoder embeddings.')

    parser.add_argument('--enc_rnn_dropout', type=float, default=0.05,
                        help='Dropout applied to encoder outputs and'
                        ' hidden states')

    parser.add_argument('--dec_emb_dropout', type=float, default=0.1,
                        help='Dropout applied to decoder embeddings.')

    parser.add_argument('--dec_rnn_dropout', type=float, default=0.05,
                        help='Dropout applied to decoder outputs and'
                        ' hidden states')

    parser.add_argument('--teacher_forcing', type=float, default=1.0,
                        help='Ratio of decoder`s own predictions and true'
                        ' target values used during training.')

    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Set encoder to compute forward and backward'
                        ' hidden states.')

    # Args set for debugging purposes.
    args = parser.parse_args([
        '--data',               'compiled/geoquery',
        '--save',               'compiled/geoquery-model',
        '--out',                'compiled/log_train.txt',
        '--epochs',             '1000',
        '--early_stop',         '500',
        '--layers',             '2',
        '--enc_hidden_size',    '128',
        '--dec_hidden_size',    '128',
        '--enc_emb_size',       '92',
        '--dec_emb_size',       '92',
        '--batch_size',         '32',
        '--teacher_forcing',    '0.8',
        '--enc_rnn_dropout',    '0.2',
        '--dec_rnn_dropout',    '0.2',
        '--enc_emb_dropout',    '0.4',
        '--dec_emb_dropout',    '0.4',
        '--mask_ratio',         '0.15'
        '--validate',
        '--bidirectional',
        '--copy',
        '--attention',
        '--best_gold'
    ])

    if validate_args(args):

        lang, datasets = io.load(args.data)
        vocab = {
            'src': Vocab(lang['vocab']['src']),
            'tgt': Vocab(lang['vocab']['tgt']),
            'stack': Vocab(lang['vocab']['stack']),
            'operator': Vocab(lang['vocab']['operator'])
        }

        log = Logger(out_path=args.out)
        line = log.add_text('')
        log.start()

        logger = {
            'log': log,
            'line': line
        }

        settings = model_settings(vocab, args)
        model = build_model(vocab, settings)

        env = {
            'model': model,
            'lang': lang
        }

        train(env, datasets)
