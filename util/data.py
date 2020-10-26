import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(batch):
    sources, targets, alignments, \
        stacks, stack_lens = zip(*batch)

    src_lens = [len(src) for src in sources]
    tgt_lens = [len(tgt) for tgt in targets]

    source_pad = pad_sequence(sources, padding_value=0)
    target_pad = pad_sequence(targets, padding_value=0)
    align_pad = pad_sequence(alignments, padding_value=0)

    max_stack_len = max(s.size(1) for s in stacks)
    max_target_len = target_pad.size(0)

    # Must be send to device.
    stack_pad = torch.zeros(
        [len(batch),
         max_target_len,
         max_stack_len]
    ).long()

    for i in range(len(batch)):
        stack = stacks[i]
        stack_pad[i, :stack.size(0), :stack.size(1)] = stack

    # Padding value is 1, for stacks that only contain 
    # start-of-sequence token. Ignored during forward pass
    # since it corresponds to decoder padding targets.
    stack_lens = pad_sequence(stack_lens, padding_value=1)
    stack_lens = stack_lens.tolist()

    return (source_pad, 
            target_pad, 
            src_lens, 
            tgt_lens, 
            align_pad,
            stack_pad,
            stack_lens)


class Dataset(Dataset):

    def __init__(self, dataset, device):
        self.data = dataset
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        x = torch.LongTensor(sample['src_i']).to(self.device)
        y = torch.LongTensor(sample['tgt_i']).to(self.device)
        align = torch.LongTensor(sample['alignment']).to(self.device)
        stacks = torch.LongTensor(sample['value_stacks']).to(self.device)
        stack_lens = torch.LongTensor(sample['stack_lens'])

        return x, y, align, stacks, stack_lens
