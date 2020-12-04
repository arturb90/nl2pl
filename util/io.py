import os
import torch

from datetime import datetime


def grammar(grammar_path):

    grammar_str = __load_file(grammar_path)
    return grammar_str


def data(
    src_train=None,
    tgt_train=None,
    src_dev=None,
    tgt_dev=None,
    src_test=None,
    tgt_test=None
):
    '''
    Loads the dataset files passed as arguments.

    :returns:   dict containing the datasets.
    '''

    datasets = {
        'train': {},
        'dev': {},
        'test': {}
    }

    if src_train:
        src_train_content = __load_file(src_train).strip()
        src_train_samples = src_train_content.split('\n')
        datasets['train'].update({
            'src': src_train_samples
        })

    if tgt_train:
        tgt_train_content = __load_file(tgt_train).strip()
        tgt_train_samples = tgt_train_content.split('\n')
        datasets['train'].update({
            'tgt': tgt_train_samples
        })

    if src_dev:
        src_dev_content = __load_file(src_dev).strip()
        src_dev_samples = src_dev_content.split('\n')
        datasets['dev'].update({
            'src': src_dev_samples
        })

    if tgt_dev:
        tgt_dev_content = __load_file(tgt_dev).strip()
        tgt_dev_samples = tgt_dev_content.split('\n')
        datasets['dev'].update({
            'tgt': tgt_dev_samples
        })

    if src_test:
        src_test_content = __load_file(src_test).strip()
        src_test_samples = src_test_content.split('\n')
        datasets['test'].update({
            'src': src_test_samples
        })

    if tgt_test:
        tgt_test_content = __load_file(tgt_test).strip()
        tgt_test_samples = tgt_test_content.split('\n')
        datasets['test'].update({
            'tgt': tgt_test_samples
        })

    if not datasets['train']:
        del datasets['train']

    if not datasets['dev']:
        del datasets['dev']

    if not datasets['test']:
        del datasets['test']

    return datasets


def load(data_path):
    '''
    Loads the language data, training and development
    datasets for training.

    :param data_path:   path containing all data.
    :returns:           language data, training and
                        development dataset.
    '''

    lang_path = f'{data_path}.lang.pt'
    train_path = f'{data_path}.train.pt'
    dev_path = f'{data_path}.dev.pt'

    lang = __load_torch(lang_path)
    train = __load_torch(train_path)

    datasets = {
        'train': train
    }

    try:
        dev = __load_torch(dev_path)
        datasets.update({
            'dev': dev
        })

    except FileNotFoundError:
        # Dev set not available.
        print(
            f'[WARN {datetime.now()}]    no development dataset'
            ' found, start training without validation.'
        )

    return lang, datasets


def __load_torch(path):
    '''
    Loads a pytorch .pt file created with torch.save.
    '''

    if not os.path.exists(path):

        raise FileNotFoundError(f'File \'{path}\' not found.')

    return torch.load(path)


def __load_file(path):
    '''
    Reads a file as string.
    '''

    content = ''
    if not os.path.exists(path):

        raise FileNotFoundError(f'File \'{path}\' not found.')

    with open(path) as file:
        content = file.read()

    return content
