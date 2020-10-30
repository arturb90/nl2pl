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

    dec_inp_size = args.dec_emb_size
    stack_align_out = None
    stack_align_in = None
    align_out = None
    align_in = None

    if args.attention or args.copy:
        enc_hid_size = args.enc_hidden_size
        align_out = args.dec_hidden_size

        if args.bidirectional:
            dec_inp_size += (enc_hid_size * 2)
            align_in = enc_hid_size * 2

        else:
            dec_inp_size += enc_hid_size
            align_in = enc_hid_size

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
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class Seq2Seq(nn.Module):

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
            # When encoder is bidirectional, use
            # forward pass hidden state as initial
            # decoder hidden state.
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

            # 0 corresponds to forward pass.
            dec_hid = enc_hid[:, 0, :, :]
            dec_cell = enc_cell[:, 0, :, :]

        else:
            dec_hid = enc_hid
            dec_cell = enc_cell

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
        src_i, src_w,
        num_parsers=1,
        beam_width=1,
        max_cycles=0
    ):

        parser = StochasticLALR(
            nlp, self.decoder,
            num_parsers=num_parsers,
            beam_width=beam_width,
            max_cycles=max_cycles
        )

        with torch.no_grad():
            src_len = [len(src_i)]
            enc_inp = src_i.unsqueeze(1)
            enc_out, enc_state = self.encoder(enc_inp, src_len)
            enc_hid = enc_state['enc_hid']
            enc_cell = enc_state['enc_cell']

            sos_token = repr(nlp.mark.out['SOS'])
            sos_token = nlp.vocab['tgt'].w2i(sos_token)
            dec_inp = torch.LongTensor((sos_token,)).to(self.device)

            if self.encoder.bidirectional:
                # When encoder is bidirectional, use
                # forward pass hidden state as initial
                # decoder hidden state.
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

                # 0 corresponds to forward pass.
                dec_hid = enc_hid[:, 0, :, :]
                dec_cell = enc_cell[:, 0, :, :]

            else:
                dec_hid = enc_hid
                dec_cell = enc_cell

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
                'enc_inp': src_w,
                'enc_out': enc_out,
                'attention': self.attention,
                'copy_attention': self.copy_attention,
                'u_align': u_align,
                'u_align_copy': u_align_copy
            }

            top, candidates = parser.parse(memory_bank)

        return top, candidates
