import random
import torch

from torch import nn

from model.modules import Encoder, Decoder, StackEncoder
from model.modules import BahdanauAttention, CopyAttention
from model.parser import StochasticLALR


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def model_settings(vocab, args):
    '''
    Builds a dictionary containing all the configuration
    settings for a sequence-to-sequence model from the
    arguments passed as training parameters.

    :param vocab:   the vocabulary corresponding to the
                    environment the model is associated
                    with.
    :param args:    the arguments passed to the training
                    script.
    :returns:       the model confoguration settings.
    '''

    dec_inp_size = args.dec_emb_size
    stack_align_out = None
    stack_align_in = None
    align_out = None
    align_in = None

    if args.attention or args.copy:
        enc_hid_size = args.enc_hidden_size
        align_out = args.dec_hidden_size

        if args.bidirectional:
            align_in = enc_hid_size * 2
            if args.attention:
                dec_inp_size += (enc_hid_size * 2)

        else:
            align_in = enc_hid_size
            if args.attention:
                dec_inp_size += enc_hid_size

    if args.stack_encoding:
        stack_hid_size = args.stack_hidden_size
        stack_align_out = args.dec_hidden_size
        stack_align_in = (stack_hid_size * 2)
        dec_inp_size += (stack_hid_size * 2)

    settings = {
        'layers': args.layers,
        'copy': args.copy,
        'attention': args.attention,
        'stack_encoding': args.stack_encoding,
        'src_vocab_size': len(vocab['src']),
        'tgt_vocab_size': len(vocab['tgt']),
        'stack_vocab_size': len(vocab['stack']),
        'enc_inp_size': args.enc_emb_size,
        'dec_inp_size': dec_inp_size,
        'enc_emb_size': args.enc_emb_size,
        'dec_emb_size': args.dec_emb_size,
        'dec_hid_size': args.dec_hidden_size,
        'enc_hid_size': args.enc_hidden_size,
        'enc_emb_drop': args.enc_emb_dropout,
        'dec_emb_drop': args.dec_emb_dropout,
        'enc_rnn_drop': args.enc_rnn_dropout,
        'dec_rnn_drop': args.dec_rnn_dropout,
        'stack_inp_size': args.stack_emb_size,
        'stack_emb_size': args.stack_emb_size,
        'stack_hid_size': args.stack_hidden_size,
        'stack_dropout': args.stack_dropout,
        'teacher_forcing': args.teacher_forcing,
        'bidirectional': args.bidirectional,
        'align_out': align_out,
        'align_in': align_in,
        'stack_align_out': stack_align_out,
        'stack_align_in': stack_align_in
    }

    return settings


def build_model(vocab, settings):
    '''
    Builds a model from a settings dictionary containing
    the model configuration data.

    :param vocab:       the vocabulary corresponding to the
                        environment the model is associated
                        with.
    :param settings:    the model configuration settings.
                        script.
    :returns:           a pytorch module instance that is
                        configured according to the model
                        settings.
    '''

    attention = None
    if settings['attention']:
        attention = BahdanauAttention(
            in_size=settings['align_in'],
            out_size=settings['align_out']
        ).to(device)

    copy_attention = None
    if settings['copy']:
        copy_attention = CopyAttention(
            in_size=settings['align_in'],
            out_size=settings['align_out']
        ).to(device)

    stack_encoder = None
    if settings['stack_encoding']:
        stack_encoder = StackEncoder(
            device, layers=1,
            vocab_size=settings['stack_vocab_size'],
            inp_size=settings['stack_inp_size'],
            emb_size=settings['stack_emb_size'],
            hid_size=settings['stack_hid_size'],
            emb_dropout=settings['stack_dropout'],
            rnn_dropout=settings['stack_dropout'],
            align_in=settings['stack_align_in'],
            align_out=settings['stack_align_out']
        ).to(device)

    encoder = Encoder(
        device,
        layers=settings['layers'],
        vocab_size=settings['src_vocab_size'],
        inp_size=settings['enc_inp_size'],
        emb_size=settings['enc_emb_size'],
        hid_size=settings['enc_hid_size'],
        emb_dropout=settings['enc_emb_drop'],
        rnn_dropout=settings['enc_rnn_drop'],
        bidirectional=settings['bidirectional']
    ).to(device)

    decoder = Decoder(
        device,
        layers=settings['layers'],
        vocab_size=settings['tgt_vocab_size'],
        inp_size=settings['dec_inp_size'],
        emb_size=settings['dec_emb_size'],
        hid_size=settings['dec_hid_size'],
        emb_dropout=settings['dec_emb_drop'],
        rnn_dropout=settings['dec_rnn_drop'],
        stack_encoder=stack_encoder
    ).to(device)

    model = Seq2Seq(
        device,
        encoder=encoder,
        decoder=decoder,
        attention=attention,
        copy_attention=copy_attention
    ).to(device)

    model.apply(__initialize)
    return model


def __initialize(model):
    # Parameter initialization.
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class Seq2Seq(nn.Module):
    '''
    The configurable main module where all threads come together.
    A Sequence-to-Sequence model according to Sutskever et al.
    https://arxiv.org/abs/1409.3215

    Consists of a encoder and a decoder recurrent neural network.
    The encoder reads in input tokens (a natural language string)
    and the decoder produces output tokens (programming language
    tokens) conditioned on the encoder context.

    May be used with content-based attention....
    (https://arxiv.org/abs/1409.0473)

    ... and copy attention pointers.
    (https://arxiv.org/abs/1606.03622)

    See 'train.py' for a full documentation of possible
    configuration options.

    :ivar device:           the device on which pytorch allocates
                            tensors. corresponds to your gpu if you
                            have a CUDA enabled gpu, otherwise cpu.
    :ivar encoder:          the encoder module associated with this
                            sequence-to-sequence model.
    :ivar decoder:          the decoder module associated with this
                            sequence-to-sequence model.
    :ivar attention:        content-based attention module over the
                            encoder hidden states.
    :ivar copy_attention:   copy attention module for generating
                            pointers over the input sequence and
                            copying the respective tokens.
    '''

    def __init__(self,
                 device,
                 encoder=None,
                 decoder=None,
                 attention=None,
                 copy_attention=None):

        super(Seq2Seq, self).__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.copy_attention = copy_attention

        # Map encoder hidden and cell dimensions
        # to decoder hidden and cell dimensions.
        dec_hid_size = self.decoder.hid_size
        enc_hid_size = self.encoder.hid_size
        if self.encoder.bidirectional:
            enc_hid_size *= 2

        self.enc2dec_hidden = nn.Linear(
            enc_hid_size,
            dec_hid_size,
            bias=True
        )

        self.enc2dec_cell = nn.Linear(
            enc_hid_size,
            dec_hid_size,
            bias=True
        )

        if self.attention:
            # Precomputing for encoder attention weights.
            out_size = self.attention.out_size
            in_size = self.attention.in_size
            self.linear_enc = nn.Linear(
                in_size,
                out_size,
                bias=True
            )

        if self.copy_attention:
            # Precomputing for encoder copy attention weights.
            out_size = self.copy_attention.out_size
            in_size = self.copy_attention.in_size
            self.linear_enc_copy = nn.Linear(
                in_size,
                out_size,
                bias=True
            )

    def forward(
        self,
        src_pad,
        tgt_pad,
        src_lens,
        tgt_lens,
        align_pad,
        stack_pad,
        stack_lens,
        teacher_forcing=0
    ):
        '''
        The forward-pass for the sequence-to-sequence model.
        Delegates to the respective submodules associated
        wit this module.

        :param src_pad:         a padded source sample batch
                                in integer representation.
        :param tgt_pad:         a padded target sample batch
                                in integer representation.
        :param src_lens:        original source sample lengths
                                before batching.
        :param tgt_lens:        original target sample lengths
                                before batching.
        :param align_pad:       the padded alignment vectors for
                                for each target sample.
        :param stack_pad:       the padded value stack contents
                                for each position in each target
                                sample.
        :param stack_lens:      the original value stack contents
                                before padding.
        :param teacher_forcing: the ratio of decoders own predictions
                                and ground truth targets fed into the
                                decoder at each decoding step.
        :returns:               all decoder outputs and copy weights
                                for each target sample in the batch.
        '''

        batch_size = tgt_pad.size(1)
        target_len = tgt_pad.size(0)
        source_len = src_pad.size(0)
        vocab_size = self.decoder.vocab_size

        dec_outs = torch.zeros(
            target_len,
            batch_size,
            vocab_size
        ).to(device)

        if self.copy_attention:
            copy_weights = torch.zeros(
                target_len,
                batch_size,
                source_len
            ).to(device)

        # Encode batch. enc_out contains unnormalized
        # encoder outputs and sequence lengths.
        enc_out, enc_state = self.encoder(src_pad, src_lens)
        enc_hid = enc_state['enc_hid']
        enc_cell = enc_state['enc_cell']
        dec_inp = tgt_pad[0, :]

        if self.encoder.bidirectional:
            # Flatten forward and backward
            # hidden and cell states.
            num_directions = 2
            enc_hid = enc_hid.view(
                self.encoder.layers,
                num_directions,
                batch_size,
                self.encoder.hid_size
            )

            enc_cell = enc_cell.view(
                self.encoder.layers,
                num_directions,
                batch_size,
                self.encoder.hid_size
            )

            enc_hid = enc_hid.transpose(1, 2)
            enc_cell = enc_cell.transpose(1, 2)
            enc_hid = enc_hid.flatten(start_dim=2, end_dim=3)
            enc_cell = enc_cell.flatten(start_dim=2, end_dim=3)

        # Map encoder hidden and cell dimensions
        # to decoder hidden and cell dimensions.
        dec_hid = self.enc2dec_hidden(enc_hid)
        dec_cell = self.enc2dec_cell(enc_cell)

        u_align = None
        if self.attention:
            # Precompute the encoder weights for attention.
            u_align = self.linear_enc(enc_out[0])

        u_align_copy = None
        if self.copy_attention:
            # Precompute the encoder weights for copy attention.
            u_align_copy = self.linear_enc_copy(enc_out[0])

        stack_inp = stack_pad[:, 0, :].to(self.device)
        stack_inp = stack_inp.transpose(0, 1)
        stack_inp_lens = stack_lens[0]

        # Decode batch, step by step.
        for i in range(1, target_len):

            dec_out, dec_state = self.decoder(
                dec_inp,
                dec_hid,
                dec_cell,
                enc_out,
                attention=self.attention,
                copy_attention=self.copy_attention,
                u_align=u_align,
                u_align_copy=u_align_copy,
                value_stacks=stack_inp,
                stack_lens=stack_inp_lens
            )

            dec_hid = dec_state['dec_hid']
            dec_cell = dec_state['dec_cell']

            dec_outs[i] = dec_out
            dec_inp = dec_out.argmax(1)

            stack_inp = stack_pad[:, i, :].to(self.device)
            stack_inp = stack_inp.transpose(0, 1)
            stack_inp_lens = stack_lens[i]

            if self.copy_attention:
                copy_weights[i] = dec_state['copy_weights']

            if self.decoder.stack_encoder:
                # If stack encodings are used,
                # we have to teacher force.
                dec_inp = tgt_pad[i]

            elif teacher_forcing:
                # Randomly choose whether to use
                # decoder predictions or ground truth
                # targets as next decoder input.
                force = random.random()
                if force < teacher_forcing:
                    dec_inp = tgt_pad[i]

        output = {'dec_outs': dec_outs}
        if self.copy_attention:
            output.update({'copy_weights': copy_weights})

        return output

    def evaluate(
        self, nlp,
        input_fields,
        num_parsers=1,
        beam_width=1,
        max_cycles=0
    ):
        '''
        The evaluation routine for the trained model. Uses a LALR(1)
        parser to enforce syntactically valid sequences during decoding,
        thus potentially correcting the decoder when it predicts a token
        that is syntactically unviable.

        :param nlp:             nl processing and parsing utils.
        :param input_fields:    fields associated with the input source
                                sequence passed during evaluation and
                                inference.
        :param num_parsers:     the maximal number of subparsers to use
                                with beam search.
        :param beam_width:      the beam width for beam search-
        :param max_cycles:      the parser, especially if poorly trained
                                may predict the same sequences over and
                                over again. define a maximum number of
                                such repeated cycles after parsing is
                                aborted.
        :returns:               the parse with highest confidence and a
                                number of other likely candidate parses
                                if beam search was used.
        '''

        parser = StochasticLALR(
            nlp, self.decoder,
            num_parsers=num_parsers,
            beam_width=beam_width,
            max_cycles=max_cycles
        )

        with torch.no_grad():
            src_i = input_fields['src_i']
            src_i = torch.LongTensor(src_i)
            src_len = [len(src_i)]

            enc_inp = src_i.unsqueeze(1).to(self.device)
            enc_out, enc_state = self.encoder(enc_inp, src_len)
            enc_hid = enc_state['enc_hid']
            enc_cell = enc_state['enc_cell']

            sos_token = repr(nlp.mark.out['SOS'])
            sos_token = nlp.vocab['tgt'].w2i(sos_token)
            dec_inp = torch.LongTensor((sos_token,)).to(self.device)

            if self.encoder.bidirectional:
                # Flatten forward and backward
                # hidden and cell states.
                num_directions = 2
                enc_hid = enc_hid.view(
                    self.encoder.layers,
                    num_directions, 1,
                    self.encoder.hid_size
                )

                enc_cell = enc_cell.view(
                    self.encoder.layers,
                    num_directions, 1,
                    self.encoder.hid_size
                )

                enc_hid = enc_hid.transpose(1, 2)
                enc_cell = enc_cell.transpose(1, 2)
                enc_hid = enc_hid.flatten(start_dim=2, end_dim=3)
                enc_cell = enc_cell.flatten(start_dim=2, end_dim=3)

            # Map encoder hidden and cell dimensions
            # to decoder hidden and cell dimensions.
            dec_hid = self.enc2dec_hidden(enc_hid)
            dec_cell = self.enc2dec_cell(enc_cell)

            u_align = None
            if self.attention:
                # Precompute the encoder weights for alignment.
                u_align = self.linear_enc(enc_out[0])

            u_align_copy = None
            if self.copy_attention:
                # Precompute the encoder weights for alignment.
                u_align_copy = self.linear_enc_copy(enc_out[0])

            memory_bank = {
                'dec_inp': dec_inp,
                'dec_hid': dec_hid,
                'dec_cell': dec_cell,
                'input_fields': input_fields,
                'enc_out': enc_out,
                'attention': self.attention,
                'copy_attention': self.copy_attention,
                'u_align': u_align,
                'u_align_copy': u_align_copy
            }

            top, candidates = parser.parse(memory_bank)

        return top, candidates
