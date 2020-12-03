
import heapq
import numpy as np
import torch

from abc import ABC, abstractmethod
from copy import deepcopy
from lark import Token


def filter_unary(nlp, tokens):
    '''
    Filters tokens that are unambiguous from a token
    sequence, that is, tokens that are the only viable
    option for the parser after parsing all tokens that
    preceed it.

    For parser states that only accept a single token from,
    the vocabulary, the decoder doesn't need to be invoked.
    Thus, in order to reflect the token sequences the decoder
    has to predict during inference in training, we filter those
    tokens out before training.

    :param nlp:     nl processing and parsing utils.
    :param tokens:  an unfiltered programming language token
                    sequence.
    :returns:       filtered token sequence.
    '''

    parser = LALR(nlp)
    filtered = [tokens[0]]

    next_ = parser.candidates
    tokeniter = iter(tokens[1:])
    for i in range(len(parser.sequence)-1):
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
        '''
        Reduces the symbols on the value and state stacks
        by the length of 'rule' and retrieves the shift action
        on the symbol corresponding to the head of 'rule'
        from the parse table.

        :param rule:    the rule to reduce by.
        '''

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
        '''
        If the expected token is the end of input marker,
        we reduce until we arrive at an end state.

        :param token:   end of input marker.
        '''

        assert token.type == '$END'

        while True:

            action, arg = self.states[self.state][token.type]
            self.reduce_(arg)

            if self.state_stack[-1] == self.end_state:
                text = ''.join(self.sequence)
                self.result = (self.value_stack[-1], text)
                self.terminated = True
                return []


class LALR(LALRBase):
    '''
    An 'stepwise' LALR parser implementation based on the
    parse tables and lexers generated by lark. A parser object
    takes in a single token and updates, the parser state.
    It returns a set of 'expected' and vatokens in that new
    parser state, which the decoder may use to perform a prediction.

    :param nlp:     nl processing and parsing utils of the
                    associated environment.
    '''

    def __init__(self, nlp):
        super(LALR, self).__init__(nlp)

        self.cache = {
            'state_stack': self.state_stack[:],
            'invalid': []
        }

        next_ = self.candidates
        # Initially, immediately parse
        # all 'expected' tokens if they are unambiguous.
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

            self.cache = {
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
            self.cache['invalid'].append(symbol)
            self.state_stack = self.cache['state_stack']

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
        num_parsers=1,
        beam_width=1,
        max_cycles=0
    ):

        super(StochasticLALR, self).__init__(nlp)

        self.decoder = decoder
        self.num_parsers = num_parsers
        self.beam_width = beam_width
        self.max_cycles = max_cycles

        self.cache = {
            'state_stack': self.state_stack[:],
            'invalid': []
        }

    def parse(self, memory_bank):
        parsepaths = [{
            'parser': LALR(self.nlp),
            'log-probabilities': [],
            'confidence': 0,
            'memory_bank': memory_bank,
            'aborted': False
        }]

        while not self.terminated:
            updated = []
            self.terminated = True

            for parsepath in parsepaths:
                subparser = parsepath['parser']

                # Search for cycles in paths and
                # and discard paths beginning the
                # 'max_cycle'-th cycle.
                if self.max_cycles \
                        and self.__cycle_detection(
                            self.max_cycles,
                            parsepath,
                            interrupt=True
                        ):

                    parsepath['aborted'] = True

                if not subparser.terminated \
                        and not parsepath['aborted']:

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
            'confidence': top['confidence'],
            'aborted': parsepath['aborted']
        }

        candidates = []
        for parsepath in parsepaths:
            candidates.append({
                'parser': parsepath['parser'],
                'confidence': parsepath['confidence'],
                'aborted': parsepath['aborted']
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
                memory_bank = parsepath['memory_bank']
                if memory_bank['copy_attention'] \
                        and symbol.type in self.nlp.OPERATOR:

                    # Set predicted index to index of predicted
                    # token from the extended vocabulary by applying
                    # the pointer operator.
                    index, token = self.nlp.OPERATOR[symbol.type].apply(
                        (memory_bank['input_fields'],
                         memory_bank['copy_weights'],
                         self.decoder.device)
                    )

                try:
                    subparser.parse(token)
                    subparser.predictions.append(index)
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

    def __cycle_detection(self, max_cycles, parsepath, interrupt=False):
        """
        The decoder, especially when it is not properly trained,
        may get stuck in a loop, repeating the same sequence over
        and over. Such repeating patterns are detected here and
        a corresponding signal is sent to the caller if the max
        number of repeats have been detected.

        :param model_cycle: the maximal number the decoder can
                            cycle through/repeat a sequence.
        :param parsepath:   the current parsepath considered.
        :param interrupt:   tries to interrupt cycle by adding
                            beginning of cycle token to list of
                            invalid tokens.
        :returns:           false if no cycle is detected, true
                            and the beginning of the repeated
                            sequence if a pattern is repeated
                            at least 'max_cycles' times.
        """

        seq = np.array(parsepath['parser'].predictions[:])
        seq_shifted = seq[:]

        start_i = 2 * max_cycles
        for i in range(start_i, len(seq)):
            # Computing "autocorrelation" between sequence
            # and shifted version of itself.
            matched = np.abs(seq[i:] - seq_shifted[:-i])

            if len(matched) > (i * max_cycles):

                # Check if the tail of the matched
                # sequence is zero.
                sos = -(i * max_cycles)
                tail = matched[sos:]
                if not np.sum(tail):

                    # Try to interrupt sequence by adding
                    # first token in repeated sequence to
                    # invalid list. Only possible if there
                    # is more than one token to choose from.
                    if self.__interrupt(seq[sos], parsepath):

                        # TODO: Implement interrupt.
                        pass

                    else:

                        # Parse has failed.
                        return True

            else:

                # No more than 'max_cycles'-1 cycles.
                break

        return False

    def __interrupt(self, index, parsepath):
        # TODO: Implement interrupt.
        return False

    def __topn_paths(self, n, parsepaths):

        # TODO: Prioritize unaborted parsepaths.

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
