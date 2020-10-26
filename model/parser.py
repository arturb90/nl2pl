import heapq
import torch

from abc import ABC, abstractmethod
from copy import deepcopy
from lark import Token


def filter_unary(nlp, tokens):
    parser = LALR(nlp)
    filtered = [tokens[0]]

    next_ = parser.candidates
    tokeniter = iter(tokens[1:])
    for i in range(len(parser.sequence)):
        next(tokeniter)

    for token in tokeniter:
        if token in next_ or \
           token.type in nlp.OPERATOR:
            filtered.append(token)
            seq_len = len(parser.sequence)
            next_ = parser.parse(token)
            diff = len(parser.sequence)-seq_len
            for i in range(diff-1):
                next(tokeniter)

    return filtered


class LALRBase(ABC):

    def __init__(self, nlp):

        self.nlp = nlp
        self.parser = nlp.lark.parser.parser.parser
        self.callbacks = self.parser.callbacks
        self.lexer = _Lexer(nlp, nlp.lexer)

        start = self.parser.parse_table.start_states.values()
        end = self.parser.parse_table.end_states.values()
        self.states = self.parser.parse_table.states
        self.start_state = list(start)[0]
        self.end_state = list(end)[0]

        self.state = self.start_state
        self.state_stack = [self.start_state]
        self.value_stack = []

        self.sequence = []
        self.predictions = []
        self.terminated = False
        self.candidates = self.lexer.next(self.state)

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    def reduce_(self, rule):
        size = len(rule.expansion)
        s = self.value_stack[-size:]
        del self.state_stack[-size:]
        del self.value_stack[-size:]

        value = self.callbacks[rule](s)
        state = self.state_stack[-1]

        name = rule.origin.name
        action, new_state = self.states[state][name]
        assert action.name == 'Shift'

        self.state_stack.append(new_state)
        self.value_stack.append(value)

    def terminate(self, token):
        while True:

            action, arg = self.states[self.state][token.type]
            self.reduce_(arg)

            if self.state_stack[-1] == self.end_state:
                text = ''.join(self.sequence)
                self.result = (self.value_stack[-1], text)
                self.terminated = True
                return []


class LALR(LALRBase):

    def __init__(self, nlp):
        super(LALR, self).__init__(nlp)

        self.__cache = {
            'state_stack': self.state_stack[:],
            'invalid': []
        }

        next_ = self.candidates
        while len(next_) == 1:
            next_ = self.parse(next_[0])

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        exclude = [
            'nlp',
            'parser',
            'callbacks',
            'states',
            'lexer'
        ]

        for k, v in self.__dict__.items():
            attr = v if k in exclude else deepcopy(v, memo)
            setattr(result, k, attr)

        return result

    def parse(self, symbol):
        next_ = self.__parse(symbol)
        while len(next_) == 1 and \
                next_[0].type not in self.nlp.OPERATOR:
            next_ = self.__parse(next_[0])
        self.candidates = next_
        return next_

    def __parse(self, symbol):
        state = self.state_stack[-1]

        try:
            action, arg = self.states[state][symbol.type]

            self.__cache = {
                'state_stack': self.state_stack[:],
                'invalid': []
            }

            if action.name == 'Shift':
                self.state_stack.append(arg)
                self.value_stack.append(symbol)
                self.state = arg

                self.sequence.append(symbol)
                next_ = self.lexer.next(arg)

            elif action.name == 'Reduce':
                self.reduce_(arg)
                next_ = self.__parse(symbol)

        except KeyError:

            # Recover from parsing error.
            self.__cache['invalid'].append(symbol)
            self.state_stack = self.__cache['state_stack']

            state = self.state_stack[-1]
            next_ = self.lexer.next(state)

            raise KeyError

        if next_ and next_[0].type == '$END':
            return self.terminate(next_[0])

        return next_


class StochasticLALR(LALRBase):

    def __init__(
        self, nlp,
        decoder,
        num_parsers=3,
        beam_width=3
    ):

        super(StochasticLALR, self).__init__(nlp)

        self.decoder = decoder
        self.num_parsers = num_parsers
        self.beam_width = beam_width

        self.__cache = {
            'state_stack': self.state_stack[:],
            'invalid': []
        }

    def parse(self, memory_bank):
        parsepaths = [{
            'parser': LALR(self.nlp),
            'log-probabilities': [],
            'confidence': 0,
            'memory_bank': memory_bank
        }]

        while not self.terminated:
            updated = []
            self.terminated = True

            for parsepath in parsepaths:
                subparser = parsepath['parser']

                if not subparser.terminated:
                    self.terminated = False
                    advanced = self.__subparse(parsepath)
                    updated.extend(advanced)

                else:
                    updated.append(parsepath)

            del parsepaths[:]
            updated = self.__topn_paths(
                self.num_parsers,
                updated
            )

            parsepaths.extend(updated)

        top = self.__topn_paths(1, parsepaths)[0]
        top = {
            'parser': top['parser'],
            'confidence': top['confidence']
        }

        candidates = []
        for parsepath in parsepaths:
            candidates.append({
                'parser': parsepath['parser'],
                'confidence': parsepath['confidence']
            })

        return top, candidates

    def __subparse(self, parsepath):
        updated = []

        memory_bank = parsepath['memory_bank']
        dec_inp = memory_bank['dec_inp'].to(self.decoder.device)

        stack_i = None,
        stack_len = None
        if self.decoder.stack_encoder:
            stack = parsepath['parser'].value_stack
            stack_seq = self.nlp.stack_sequence(stack, filter_token=True)
            stack_i = self.nlp.stack2indices(stack_seq, delimiters=True)
            stack_i = torch.LongTensor(stack_i).to(self.decoder.device)
            stack_i = stack_i.unsqueeze(1)
            stack_len = [len(stack_i)]

        dec_out, dec_state = self.decoder(
            dec_inp,
            memory_bank['dec_hid'],
            memory_bank['dec_cell'],
            memory_bank['enc_out'],
            memory_bank['attention'],
            memory_bank['copy_attention'],
            memory_bank['u_align'],
            memory_bank['u_align_copy'],
            stack_i, stack_len
        )

        memory_bank['dec_hid'] = dec_state['dec_hid']
        memory_bank['dec_cell'] = dec_state['dec_cell']

        if memory_bank['copy_attention']:
            memory_bank['copy_weights'] = dec_state['copy_weights']

        scores = dec_out
        next_ = parsepath['parser'].candidates
        updated = self.__advance(
            parsepath,
            next_,
            scores,
            self.beam_width
        )

        return updated

    def __advance(self, parsepath, next_, scores, beam_width):
        updated = []

        while beam_width > 0 and next_:
            predictions = self.__pick(next_, scores, beam_width)

            for p in predictions:
                copy = deepcopy(parsepath)
                subparser = copy['parser']

                index = p[0].item()
                symbol = self.nlp.indices2tokens([index])[0]
                token = symbol

                # Resolve operator symbol if copy attention is used.
                # TODO: Enforce usage of copy attention.
                memory_bank = parsepath['memory_bank']
                if memory_bank['copy_attention'] \
                        and symbol.type in self.nlp.OPERATOR:
                    operator = self.nlp.OPERATOR[symbol.type]
                    copy_weights = memory_bank['copy_weights']
                    enc_inp = memory_bank['enc_inp']
                    _, indices = torch.topk(copy_weights, len(enc_inp))
                    indices = indices.squeeze()
                    copy_w = enc_inp[indices[0]]
                    while not operator.target.match(copy_w):
                        indices = indices[1:]
                        copy_w = enc_inp[indices[0]]
                    token = Token(symbol.type, copy_w)

                try:
                    subparser.parse(token)
                    subparser.predictions.append(p[0].item())
                    i = next_.index(symbol)
                    beam_width -= 1
                    next_.pop(i)

                    probs = copy['log-probabilities']
                    confidence = copy['confidence']
                    probs.append(p[1].item())
                    confidence = sum(probs) / len(probs)

                    memory_bank = copy['memory_bank']
                    memory_bank['dec_inp'] = p[0].unsqueeze(0)
                    copy['log-probabilities'] = probs
                    copy['confidence'] = confidence
                    updated.append(copy)

                except KeyError:
                    i = next_.index(symbol)
                    next_.pop(i)

        return updated

    def __topn_paths(self, n, parsepaths):

        def predicate(i):
            return i['confidence']

        topn = heapq.nlargest(n, parsepaths, key=predicate)
        return topn

    def __pick(self, candidates, scores, beam_width):
        squeezed = scores.squeeze()
        indices = self.nlp.tokens2indices(candidates)
        indices = torch.LongTensor(indices)
        map_ = map(lambda i: (i, squeezed[i]), indices)

        predictions = heapq.nlargest(
            beam_width,
            map_,
            key=lambda i: i[1]
        )

        return predictions


class _Lexer:

    def __init__(self, nlp, lexer):
        self.nlp = nlp
        self.lexer = lexer

    def next(self, state):
        next_ = []
        lexer = self.lexer.lexers[state]

        if lexer.mres:
            regex, types = lexer.mres[0]

            for type_ in types.values():
                tokens = self.nlp.match_tokens(regex, type_)
                next_.extend(tokens)

        else:
            # If the parser has no more options, we have
            # reached an end state and complete the parse.
            next_ = [Token('$END', '')]

        return next_
