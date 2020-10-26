import torch

from torch import nn
from torch.functional import F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNN(nn.Module):

    def __init__(self,
                 device,
                 layers=1,
                 vocab_size=None,
                 inp_size=128,
                 emb_size=128,
                 hid_size=256,
                 emb_dropout=0.0,
                 rnn_dropout=0.0,
                 bidirectional=False):

        super(RNN, self).__init__()

        self.device = device
        self.vocab_size = vocab_size

        self.layers = layers
        self.inp_size = inp_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.emb_dropout = emb_dropout
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)

        self.LSTM = nn.LSTM(
            input_size=inp_size,
            hidden_size=hid_size,
            num_layers=layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional
        ).to(device)

        def forward(self):
            raise NotImplementedError


class Encoder(RNN):

    def __init__(self,
                 device,
                 layers=1,
                 vocab_size=None,
                 inp_size=128,
                 emb_size=128,
                 hid_size=256,
                 emb_dropout=0.0,
                 rnn_dropout=0.0,
                 bidirectional=False):

        super(Encoder, self).__init__(
            device,
            layers=layers,
            vocab_size=vocab_size,
            inp_size=inp_size,
            emb_size=emb_size,
            hid_size=hid_size,
            emb_dropout=emb_dropout,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional
        )

    def forward(self, src_pad, src_lens):
        embedded = self.embedding(src_pad)

        # inp           [ _inp_len : _batch_size ]
        # embedded      [ _inp_len : _batch_size : _emb_dim]
        emb_dropout = self.emb_dropout(embedded)

        src_packed = pack_padded_sequence(
            emb_dropout,
            src_lens,
            enforce_sorted=False,
        )

        out_packed, (enc_hid, enc_cell) = self.LSTM(src_packed)

        enc_out = pad_packed_sequence(
            out_packed
        )

        enc_hid = self.rnn_dropout(enc_hid)
        enc_cell = self.rnn_dropout(enc_cell)

        enc_state = {
            'enc_hid': enc_hid,
            'enc_cell': enc_cell
        }

        return enc_out, enc_state


class Decoder(RNN):

    def __init__(self,
                 device,
                 layers=1,
                 vocab_size=None,
                 inp_size=128,
                 emb_size=128,
                 hid_size=256,
                 emb_dropout=0.0,
                 rnn_dropout=0.0,
                 stack_encoder=None):

        super(Decoder, self).__init__(
            device,
            layers=layers,
            vocab_size=vocab_size,
            inp_size=inp_size,
            emb_size=emb_size,
            hid_size=hid_size,
            emb_dropout=emb_dropout,
            rnn_dropout=rnn_dropout,
            bidirectional=False
        )

        self.out = nn.Linear(hid_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.stack_encoder = stack_encoder

    def forward(self,
                dec_inp,
                hid_prev,
                cell_prev,
                enc_out,
                attention=None,
                copy_attention=None,
                u_align=None,
                u_align_copy=None,
                value_stacks=None,
                stack_lens=None):

        dec_inp = dec_inp.unsqueeze(0)
        embedded = self.embedding(dec_inp)
        emb_dropout = self.emb_dropout(embedded)
        context = emb_dropout

        dec_state = {}
        enc_out, src_lens = enc_out
        if attention:
            # Compute context vector and attention weights.
            assert u_align is not None
            attn_h, attn_weights = attention(
                hid_prev,
                enc_out,
                src_lens,
                u_align
            )

            context = torch.cat((attn_h, context), dim=2)
            dec_state.update({'attn_weights': attn_weights})

        if self.stack_encoder:
            stack_attn, _ = self.stack_encoder(
                hid_prev,
                value_stacks,
                stack_lens
            )

            context = torch.cat((stack_attn, context), dim=2)

        if copy_attention:
            # Compute copy attention weights
            assert u_align_copy is not None
            copy_weights = copy_attention(
                hid_prev,
                enc_out,
                src_lens,
                u_align_copy
            )

            # context = torch.cat((attn_h, emb_dropout), dim=2)
            # dec_state.update({'attn_weights': attn_weights})
            dec_state.update({'copy_weights': copy_weights})

        dec_out, (dec_hid, dec_cell) = self.LSTM(
            context, (hid_prev, cell_prev)
        )

        dec_hid = self.rnn_dropout(dec_hid)
        dec_cell = self.rnn_dropout(dec_cell)

        dec_out = self.out(dec_out.squeeze(0))
        dec_out = self.log_softmax(dec_out)

        dec_state.update({
            'dec_hid': dec_hid,
            'dec_cell': dec_cell
        })

        return dec_out, dec_state


class StackEncoder(RNN):

    def __init__(self,
                 device,
                 layers=1,
                 vocab_size=None,
                 inp_size=128,
                 emb_size=128,
                 hid_size=256,
                 emb_dropout=0.0,
                 rnn_dropout=0.0,
                 align_in=128,
                 align_out=128):

        super(StackEncoder, self).__init__(
            device,
            layers=layers,
            vocab_size=vocab_size,
            inp_size=inp_size,
            emb_size=emb_size,
            hid_size=hid_size,
            emb_dropout=emb_dropout,
            rnn_dropout=rnn_dropout,
            bidirectional=True,
        )

        self.align_in = align_in
        self.align_out = align_out

        self.attention = BahdanauAttention(
            align_in, align_out
        )

        self.linear_enc = nn.Linear(
            align_in,
            align_out,
            bias=True
        )

    def forward(self, query, src_pad, src_lens):
        embedded = self.embedding(src_pad)
        emb_dropout = self.emb_dropout(embedded)

        src_packed = pack_padded_sequence(
            emb_dropout,
            src_lens,
            enforce_sorted=False,
        )

        out_packed, (enc_hid, enc_cell) = self.LSTM(src_packed)

        enc_out = pad_packed_sequence(
            out_packed
        )

        enc_out, src_lens = enc_out
        u_align = self.linear_enc(enc_out)
        attn_h, attn_weights = self.attention(
            query,
            enc_out,
            src_lens,
            u_align
        )

        return attn_h, attn_weights


class BaseAttention(nn.Module):

    def __init__(
        self,
        in_size,
        out_size
    ):

        super(BaseAttention, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.linear_q = nn.Linear(out_size, out_size, bias=True)
        self.linear_v = nn.Linear(out_size, 1, bias=False)

    def forward(self, query, enc_out, src_lens, u_align):
        # Create a mask from source lengths
        # to zero padding indices out during
        # batch processing.
        max_len = enc_out.size(0)
        arange = torch.arange(max_len)[None, :]
        mask = arange < src_lens[:, None]
        mask = mask.transpose(0, 1)

        # Compute alignment scores
        layers = query.size(0)
        top_layer = query[layers-1, :, :]
        top_layer = top_layer.unsqueeze(0)
        w_align = self.linear_q(top_layer)
        tanh_align = torch.tanh(w_align + u_align)
        score_align = self.linear_v(tanh_align)
        score_align = score_align.squeeze(2)
        score_align[~mask] = float('-inf')

        # Compute attention weights.
        attn_weights = F.softmax(score_align, dim=0)
        attn_weights = attn_weights.unsqueeze(2)

        return attn_weights


class BahdanauAttention(BaseAttention):

    def __init__(
        self,
        in_size,
        out_size
    ):

        super(BahdanauAttention, self).__init__(
            in_size, out_size
        )

    def forward(self, query, enc_out, src_lens, u_align):

        attn_weights = super().forward(
            query, enc_out, src_lens, u_align
        )

        # Multiply attention weights with encoder
        # outputs to obtain context vectors.
        attn_weights = attn_weights.transpose(0, 1)
        attn_weights = attn_weights.transpose(1, 2)
        enc_out = enc_out.transpose(0, 1)
        attn_h = torch.bmm(attn_weights, enc_out)
        attn_h = attn_h.transpose(0, 1)

        attn_weights = attn_weights.squeeze(1)
        attn_weights = attn_weights.transpose(0, 1)

        return attn_h, attn_weights


class CopyAttention(BaseAttention):

    def __init__(
        self,
        in_size,
        out_size
    ):

        super(CopyAttention, self).__init__(
            in_size, out_size
        )

    def forward(self, query, enc_out, src_lens, u_align):

        attn_weights = super().forward(
            query, enc_out, src_lens, u_align
        )

        attn_weights = attn_weights.squeeze(2)
        attn_weights = attn_weights.transpose(0, 1)
        return attn_weights
