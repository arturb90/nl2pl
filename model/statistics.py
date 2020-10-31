import numpy as np
import torch


class Scorer:

    def __init__(self, nlp, vocab):
        self.nlp = nlp
        self.vocab = vocab

        self.__total_examples = 0
        self.__total_tokens = 0

        self.__correct_examples = 0
        self.__correct_tokens = 0

        self.__aborted_parses = 0

    def score(self, results, tgt_seq):

        if results['copy_used']:
            tgt_seq = self.__resolve_alignment(
                tgt_seq,
                results['alignment']
            )

        pred_seq = results['predictions']
        tgt_len = len(tgt_seq)
        pred_len = len(pred_seq)

        if tgt_len < pred_len:
            pred_seq = pred_seq[:tgt_len]

        elif tgt_len > pred_len:
            pad_token = self.nlp.mark.out['PAD']
            pad_i = self.nlp.tokens2indices([pad_token])[0]
            pred_seq = [*pred_seq, *([pad_i] * tgt_len)]
            pred_seq = pred_seq[:tgt_len]

        equal = [
            True if i == j else False
            for i, j in zip(pred_seq, tgt_seq)
        ]

        self.__total_tokens += tgt_len
        self.__total_examples += 1
        self.__correct_tokens += sum(equal)
        self.__correct_examples += all(i is True for i in equal)
        self.__aborted_parses += results['aborted']

    def results(self):
        return {
            'accuracy': self.__correct_tokens / self.__total_tokens,
            'gold_acc': self.__correct_examples / self.__total_examples,
            'aborted': self.__aborted_parses
        }

    def __resolve_alignment(self, tgt_seq, alignment):

        tgt_vec = np.array(tgt_seq)
        align_vec = np.array(alignment[1:])
        tgt_vocab_len = len(self.vocab['tgt'])

        fn = (lambda a: a == 0)
        result = np.where(
            fn(align_vec), tgt_vec,
            align_vec + tgt_vocab_len
        )

        return result.tolist()


class Statistics:

    def __init__(self, loss, batch_count, results):

        total_tokens = 0
        total_samples = 0
        correct_tokens = 0
        correct_samples = 0

        for i in range(batch_count):
            len_ = results[i]['tgt_len']
            preds = results[i]['predictions']
            targets = results[i]['targets']

            if results[i]['copy_attn_used']:
                copy_preds = results[i]['copy_predictions']
                copy_tgts = results[i]['copy_targets']

            # Set indices to zero where target is zero.
            indices = preds.argmax(1)
            indices = torch.where(targets == 0, targets, indices)

            if results[i]['copy_attn_used']:
                # If copy attention is used, replace operator indices
                # with the predicted index in the extended vocabulary,
                # but only where an operator's index was actually predicted
                _, copy_indices = copy_preds.topk(1, dim=1)
                copy_indices = copy_indices.squeeze(1)
                copy_indices = torch.where(
                    copy_tgts == 0, copy_tgts, copy_indices
                )

                # TODO: Set copy targets and copy indices to zero
                # where no operator token was predicted.

                tgt_vocab_size = results[i]['tgt_vocab_size']
                copy_tgts = copy_tgts + tgt_vocab_size
                targets = torch.where(
                    copy_tgts == tgt_vocab_size,
                    targets,
                    copy_tgts
                )

                copy_indices = copy_indices + tgt_vocab_size
                indices = torch.where(
                    copy_indices == tgt_vocab_size,
                    indices,
                    copy_indices
                )

            # Add number of nonzero targets.
            nonzero_tgt = targets[targets.nonzero()].squeeze()
            total_tokens += nonzero_tgt.size(0)

            # Determine the number of correctly predicted tokens.
            nonzero_idx = indices[indices.nonzero()].squeeze()
            nonzero_equal = torch.eq(nonzero_tgt, nonzero_idx)
            num_equal_tokens = torch.sum(nonzero_equal)
            correct_tokens += num_equal_tokens.item()

            # Determine the number of exact matches.
            equal = torch.eq(targets, indices)
            samplewise = equal.view(-1, len_)
            total_samples += samplewise.size(0)
            for i in range(samplewise.size(0)):
                sample = samplewise[i, :]
                if torch.all(sample):
                    correct_samples += 1

        self.loss = loss / batch_count
        self.accuracy = correct_tokens / total_tokens
        self.gold_accuracy = correct_samples / total_samples
