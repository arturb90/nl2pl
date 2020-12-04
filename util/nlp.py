'''
Main helper classes and functions for processing
and preparing the grammar, inputs to and outputs
from the neural network.
'''

import abc
import re
import string
import torch

from collections import defaultdict
from lark import Tree, Token


def parse_grammar(grammar_content):
    '''
    Extracts operator terminals from a grammar file
    and rewrites them such that the grammar file is
    is a valid lark grammar.

    :param grammar_content: the 'augmented' grammar file
                            including operator terminals.
    '''

    lines = grammar_content.split('\n')
    operators = []
    filtered = []

    for line in lines:

        # Filter operator terminals such that they are
        # 'normal' terminals again.
        if re.match(r'^(\*|~|@|\*@)[A-Z]*\s*:', line):
            l, operator = __parse_line(line)
            operators.append(operator)
            filtered.append(l)

        else:
            filtered.append(line)

    filtered = '\n'.join(filtered)
    return filtered, operators


def __parse_line(line):
    '''
    Helper for the function 'parse_grammar'.
    Filters the line and removes operator
    prefixes (if there are any)from lines
    containing terminals.

    :param line:    a line in a lark grammar file.
    '''

    split = line.split(':')
    lhs = split[0].strip()
    rhs = split[1].strip()

    op, name = lhs[0], lhs[1:]
    filtered = f'{name}\t:\t{rhs}'

    args = []
    items = rhs.split(' ')
    for item in items:

        if re.match(r'\".*\"', item):
            args.append(item[1:-1])

        elif re.match(r'/.*/', item):
            args.append(item)

    operator = {
        'name': name,
        'type': op,
        'args': args
    }

    return filtered, operator


class NLP:
    '''
    The main helper class for processing and parsing
    inputs to and outputs from the neural network.
    It is initializied from a lark parser each time
    a model trained or loaded for inference.

    :param lark:        the lark parser generated from
                        a grammar file.
    :param operators:   all operator tokens in the model
                        environment this instance belongs
                        to.

    :ivar NONTERMINALS:     collection of non-terminal
                            symbols belonging to this parsing
                            environment.
    :ivar TERMINALS:        collection if terminal symbols.
    :ivar OPERATOR:         collection of operator terminals.
    :ivar TOKENS:           collection of all tokens apperaing
                            in the target datasets passed during
                            preprocessing.
    '''

    class mark:
        '''
        Special symbols for both input and outputs.

        <PAD>: Padding token.
        <SOS>: Start of sequence.
        <EOS>: End of sequence.
        <UNK>: Unknown token.
        '''

        inp = {
            'PAD': '<PAD>',
            'SOS': '<SOS>',
            'EOS': '<EOS>',
            'UNK': '<UNK>'
        }

        out = {
            'PAD': Token('<PAD>', '<PAD>'),
            'SOS': Token('<SOS>', '<SOS>'),
            'UNK': Token('<UNK>', '<UNK>'),
            'EOS': Token('<EOS>', '<EOS>')
        }

    def __init__(self, lark, operators):
        self.lark = lark
        self.lexer = lark.parser.lexer

        self.NONTERMINALS = None
        self.TERMINALS = None
        self.OPERATOR = None
        self.TOKENS = None

        self.vocab = None

        # Initializes NONTERMINAL, TERMINAL, OPERATOR and
        # TOKEN fields.
        self.__collect_symbols(lark, operators, self.lexer)

    def normalize(self, text, delimiters=False, lower=True):
        '''
        Normalizes an input text, pads punctuation in
        the text.

        :param text:        the text to normalize.
        :param delimiters:  whether to insert sequence delimiters
                            '<SOS>' and '<EOS>'.
        :returns:           list of all white-space separated
                            tokens in the text.
        '''

        text = self.__pad_punctuation(text)

        if lower:
            text = text.strip().lower()

        tokens = text.split()

        if delimiters:
            sos_token = self.mark.inp['SOS']
            eos_token = self.mark.inp['EOS']
            tokens = [sos_token, *tokens, eos_token]

        return tokens

    def tokenize(self, text, delimiters=False):
        '''
        Parses a target program and extracts the token
        sequence from the parse tree.

        :param text:        the input program string.
        :param delimiters:  whether to use an '<SOS> indicator.
        :returns:           token sequence corresponding to the
                            input program as list.
        '''

        parse = self.lark.parse(text)
        tokens = self.token_sequence(parse)

        if delimiters:
            sos_token = self.mark.out['SOS']
            tokens = [sos_token, *tokens]

        return tokens

    def indices2tokens(self, indices):
        '''
        Converts a token sequence represented as integers into
        it's original representation.

        :param indices:     the indices of each token in the
                            sequence according to the target
                            vocabulary.
        :returns:           sequence of programming language
                            tokens.
        '''

        tokens = [self.vocab['tgt'].i2w(i) for i in indices]

        try:
            tokens = [self.TOKENS[t] for t in tokens]

        except KeyError as e:
            print(e)
            return []

        return tokens

    def tokens2indices(self, tokens):
        '''
        Converts a token sequence into an integer representation
        defined by the respective indices in the target vocabulary.

        :param tokens:      the token sequence.
        :returns:           integer representation of the token
                            sequence.
        '''

        indices = []

        for token in tokens:

            try:

                repr_ = repr(token)
                index = self.vocab['tgt'].w2i(repr_)
                indices.append(self.vocab['tgt'].w2i(repr_))

            except KeyError:

                # TODO: Useless?
                unk_token = repr(self.mark.out['UNK'])
                index = self.vocab['tgt'].w2i(unk_token)
                indices.append(index)

        return indices

    def stack2indices(self, stack, delimiters=False):
        '''
        Converts a symbol sequence in a flattened parser stack into
        an integer representation defined by the respective indices
        in the stack vocabulary.

        :param stack:       the symbols sequence corresponding
                            to a value stack
        :param delimiters:  whether to insert a '<SOS>' token at
                            the beginning of the sequence.
        :returns:           integer representation of the symbol
                            sequence.
        '''

        if delimiters:
            sos_token = repr(self.mark.out['SOS'])
            indices = [self.vocab['stack'].w2i(sos_token)]
        else:
            indices = []

        for symbol in stack:
            repr_ = repr(symbol)
            indices.append(self.vocab['stack'].w2i(repr_))

        return indices

    def stack_sequence(self, stack, filter_token=False):
        '''
        Flattens a parser value stack consisting of partial parse-
        trees and tokens and optionally filters tokens out.

        :param stack:           the original parser value stack.
        :param filter_token:    whether to filter tokens on the stack.
        :returns:               a flattened sequence of symbols as they
                                appear on the value stack.
        '''

        result = []

        for item in stack:

            if type(item) is Token:
                if filter_token:
                    continue
                result.append(item)

            elif type(item) is Tree:
                # Flatten the root tree node into a sequence
                # of nonterminal symbols in the tree.
                nt_seq = self.nonterminal_sequence(item)
                result.extend(nt_seq)

        return result

    def nonterminal_sequence(self, tree):
        '''
        Flattens a tree on the parser value stack corresponding to
        an non-terminal symbol. Converts the tree into a sequence
        of non-terminal symbols depth-first, from left to right.

        :param tree:    the tree node to linearize.
        :returns:       a depth-first, left to right sequence
                        of non-terminal tree nodes.
        '''

        nt = self.NONTERMINALS[tree.data].nt
        result = [nt]

        for i in range(len(tree.children)):

            if isinstance(tree.children[i], Tree):
                nt = self.nonterminal_sequence(tree.children[i])
                result.extend(nt)

            # TODO: Include tokens?
            # elif isinstance(tree.children[i], Token):
            #     result.append(tree.children[i])

        return result

    def token_sequence(self, tree):
        '''
        Extracts the  ordered token sequence from a parse
        tree (the 'leaves' of the tree).

        :param tree:    the input parse tree.
        :returns:       ordered token sequence.
        '''

        result = []

        for i in range(len(tree.children)):

            if isinstance(tree.children[i], Tree):
                # Descend recursively.
                tokens = self.token_sequence(tree.children[i])
                result.extend(tokens)

            elif isinstance(tree.children[i], Token):
                result.append(tree.children[i])

        return result

    def alignment(self, source, target, sample_vocab):
        '''
        Any target token that has a pointer terminal type is
        is assumed to appear in the input sequence and mapped
        to the corresponding position in the input sequence.

        For example, if a the string 'berlin' appears in the
        target program and strings are defined in the grammar
        to be pointer operators, we search for the word in the
        input sequence this target sample belongs to. If it is
        found we record the input position in an alignment vector.

        The alignments are used during training to learn which input
        position pointer operator tokens should copy the token values
        from.

        :param source:          source input string.
        :param target:          target output program.
        :param sample_vocab:    a small source sample vocab, taking
                                tokens in the input sentence to integer
                                indices.
        '''

        # Map target tokens that are of operator type
        # to words in the input vocabulary.
        alignment = []

        for token in target:

            if token.type in self.OPERATOR:
                operator = self.OPERATOR[token.type]
                match = operator.source.findall(token.value)
                word = match[0]

                try:
                    tgt_in_src = sample_vocab['w2i'][word]
                    alignment.append(tgt_in_src)

                except KeyError:
                    alignment.append(0)

            else:
                alignment.append(0)

        return alignment

    def match_tokens(self, regex, type_=None):
        '''
        Return token values in vocabulary that match a
        regular expression. Optionally, filter by type.

        :param regex:   the regex to be matched.
        :param type_:   the token type to filter.
        :returns:       matched token values.
        '''

        find_type = (lambda t: t.type == type_)
        find_val = (lambda t: regex.match(t.value))
        find_tokens = (lambda t: find_type(t) and find_val(t))

        if type_ in self.OPERATOR:
            operator = self.OPERATOR[type_]
            result = operator.tokens

        elif type_:
            result = list(filter(find_tokens, self.TOKENS.values()))

        else:
            result = list(filter(find_val, self.TOKENS.values()))

        return list(result)

    def collect_tokens(self, vocab):
        '''
        Collects all tokens as they appear in the datasets
        provided during preprocessing and as they are defined
        in the grammar this nlp object is associated with.
        Also associates terminal opjects with the tokens of the
        respective terminal type.

        :param vocab:   the vocabulary associated with this
                        nlp object and environment.
        :returns:       dictionary of tokens.
        '''

        terminals = {}

        for terminal in self.TERMINALS.values():
            if not terminal.tokens:
                terminals[terminal.name] = terminal

        for token in self.mark.out.values():
            self.TOKENS[repr(token)] = token

        for operator in self.OPERATOR.values():
            # Add special token for operator terminals.
            token = Token(operator.name, f'<{operator.type}>')
            self.TOKENS[repr(token)] = token
            operator.tokens.append(token)

        for item in vocab['tgt']._w2i.keys():
            type_ = re.match(r'^[A-Za-z]*', item).group(0)

            if type_ == 'Token':
                value = re.findall(r'\(.*\)$', item)[0].split(', ', 1)
                token_type = value[0][1:]
                token_value = value[1][1:-2]

                if token_type in terminals and \
                   token_type not in self.OPERATOR:
                    token = Token(token_type, token_value)
                    # Add token to the list terminal holds of
                    # tokens that exist of this type.
                    terminals[token.type].tokens.append(token)
                    self.TOKENS[repr(token)] = token

    def __pad_punctuation(self, text):
        '''
        Pads punctuation in 'text'
        A string "For example, this text." is converted
        to "For example , this text ."
        '''

        punct = string.punctuation
        mapping = {key: " {0} ".format(key) for key in punct}
        translator = str.maketrans(mapping)
        padded = text.translate(translator)
        return padded

    def __collect_symbols(self, parser, operators, lexer):
        '''
        Builds the dicts for TOKEN, OPERATOR, TERMINAL and
        NONTERMINAL attributes this class uses to perform it's
        operations.
        '''

        tokens = {}
        operator = {}
        terminals = {}
        nonterminals = {}

        ops = self.__build_operators(operators)

        for terminal in parser.terminals:

            if terminal.name in ops:
                op = ops[terminal.name]
                t = TerminalOp(terminal, operator=op)
                assert t.pattern.type == 're'
                operator[f'{t.name}'] = t
                t.parse_args()

            else:
                t = Terminal(terminal)

            terminals[t.name] = t
            tokens.update({
                repr(token): token
                for token in t.tokens
            })

        expansions = defaultdict(list)
        for rule in parser.rules:
            e = rule.expansion
            expansions[rule.origin].append(e)

        for k, v in expansions.items():
            nt = NonTerminal(k, v)
            nonterminals[k.name] = nt

        self.NONTERMINALS = nonterminals
        self.TERMINALS = terminals
        self.OPERATOR = operator
        self.TOKENS = tokens

    def __build_operators(self, operators):
        '''
        Links each operator terminal extracted from the grammar
        to the operations they correspond to.

        :param operators:   the operator data extracted from the
                            grammar.
        :returns:           instantiated operator functions to
                            be applied on occurence of a particular
                            operator in the output token stream.
        '''

        result = {}

        # The functions to be applied on occurence
        # of the respective operator in the token stream.
        op_def = {
            '*': (lambda t, args: self.__STAR(t, args)),
            '@': (lambda t, args: self.__ANON(t, args)),
            '#': (lambda t, args: self.__HASH(t, args))
        }

        for op in operators:
            type_ = op["type"]
            name = op['name']
            args = op['args']

            op_applied = (
                lambda t, args: op_def[t.type[:1]](t, args)
            )

            result.update({f'{name}': [type_, args, op_applied]})

        return result

    def __STAR(self, t, args):
        '''
        The star ('*') operator corresponds to a pointer to the
        input sequence. Whenever a token of pointer/star type is
        predicted this function is applied to the token to copy
        it's value from the input sequence.

        :param operators:   the operator terminal object.
        :param args:        list containing the input fields (0)
                            created during preprocessing or inference
                            and the copy probabilites, that indicate
                            which element in the input sequence sould
                            be copied for this output token.
        :returns:           resolved output token where the token value
                            corresponds to the token copied from the
                            input sequence.
        '''

        input_fields = args[0]
        copy_weights = args[1]
        device = args[2]

        src_t = self.normalize(input_fields['src'], lower=False)
        copy_weights = copy_weights[:, 1:-1].to(device)

        shape = (1, copy_weights.shape[1])
        val_buffer = torch.empty(shape)
        idx_buffer = torch.empty(
            shape,
            dtype=torch.long
        ).to(device)

        torch.topk(
            copy_weights, len(src_t),
            out=(val_buffer, idx_buffer)
        )

        tgt_vocab_len = len(self.vocab['tgt'])
        input_vocab = Vocab(input_fields['sample_vocab'])
        extended_vocab = self.vocab['tgt'].extend(input_vocab)
        idx_buffer = idx_buffer[0].tolist()
        copy_t = f'<{t.type}>'

        if idx_buffer:
            copy_t = src_t[idx_buffer[0]]
            copy_i = extended_vocab.w2i(copy_t)
            idx_buffer = idx_buffer[1:]

        while not t.target.match(copy_t):
            # If selected word does not match the regex
            # defined by the operator, try the noxt most
            # likely copy token.

            if idx_buffer:
                copy_t = src_t[idx_buffer[0]]
                copy_i = tgt_vocab_len + input_vocab.w2i(copy_t)
                idx_buffer = idx_buffer[1:]

            else:
                # Fallback if no viable token found.
                copy_t = f'<{t.type}>'
                break

        token = Token(t.name, copy_t)
        # Remove input tokens from target vocab.
        self.vocab['tgt'].remove(input_vocab)
        return copy_i, token

    def __ANON(self, t, args):

        # TODO: Implement.
        src, dst = args
        out_src = src
        out_dst = dst

        out_src = t.source.sub(f'<{t.name}>', src)
        _, out_dst = self.__STAR(t, src, dst)

        return out_src, out_dst

    def __HASH(self, t, args):

        # TODO: Implement.
        src, dst = args
        out_src = src
        out_dst = dst

        return out_src, out_dst


class Symbol(abc.ABC):
    '''
    Base symbol class.
    '''

    def __init__(self, name, type_):
        self.name = name
        self.type = type_


class Terminal(Symbol):
    '''
    A terminal symbol.

    :ivar tdef:     The lark terminal definition.
    :ivar pattern:  The regex pattern for tokens that constitute
                    terminals of this type.
    :ivar tokens:   list of all tokens explicity appearing as
                    alternative in the terminals regex or that appear
                    in the datasets that this terminal instance belongs to.
    '''

    def __init__(self, tdef):
        super(Terminal, self).__init__(tdef.name, tdef.name)

        self.tdef = tdef
        self.pattern = tdef.pattern
        self.tokens = self.__resolve(tdef.name, tdef.pattern)

    def __resolve(self, name, pattern):
        split = []
        tokens = []

        if pattern.value.startswith('(?:'):
            replaced = re.sub(r'[\(?:\)\\]', '', pattern.value)
            split = replaced.split('|')
            self.pattern.type = 'str'

        if split:
            tokens = [Token(name, value) for value in split]

        elif self.pattern.type == 'str':
            tokens = [Token(name, pattern.value)]

        return tokens

    def __repr__(self):
        return f'Terminal(\'{self.type}\')'


class TerminalOp(Terminal):
    '''
    A terminal symbol with an function associated with it
    that is applied to each token of this terminal type.

    :ivar operator:     the operator data.
    :ivar type:         the operator terminal type.
    :ivar op_def:       the function to apply to tokens of
                        this operator terminal type.
    :ivar source:       the terminal regex in terms of the
                        input source tokens. For example, a string
                        includes double quotes in a program. when
                        we want to copy a string from the input sequence
                        we omit the double quotes when matching input
                        tokens.
    :ivar target:       the terminal target regex.
    '''

    def __init__(self, tdef, operator):
        super(TerminalOp, self).__init__(tdef)
        self.operator = operator
        self.type = f'{operator[0]}{tdef.name}'
        self.args = operator[1]
        self.op_def = operator[2]
        self.name = tdef.name

        self.source = None
        self.target = None

    def apply(self, args):
        return self.op_def(self, args)

    def parse_args(self):
        source = []
        target = []

        for arg in self.args:
            part = arg
            if re.match(r'/.*/', arg):
                part = arg[1:-1]
                source.append(part)
            target.append(part)

        self.source = re.compile(''.join(source))
        self.target = re.compile(''.join(target))

    def convert(self, token):
        # Get the representation of the token in terms
        # of the input text, if there is one, with respect
        # to this terminals regex definitions.
        if self.source is None or self.target is None:
            self.parse_args()

        matches = self.source.findall(token.value)

        text = None
        if matches:
            text = matches[0]

        return text


class NonTerminal(Symbol):
    '''
    A nonterminal symbol.

    :ivar nt:           larks non-terminal definiton.
    :ivar expansions:   the expansion rules associated
                        with this nonterminal.
    '''

    def __init__(self, nt, expansions):
        super(NonTerminal, self).__init__(nt.name, nt.name)

        self.nt = nt
        self.expansions = expansions


class Vocab:
    '''
    The primary vocabulary data structure used to convert
    from tokens to indices and vice versa.

    :param vocab:   dictionary containing i2w and w2i
                    dictionaries corresponding to some
                    mapping of indices to words/tokens.
    '''

    def __init__(self, vocab):
        self._i2w = vocab['i2w']
        self._w2i = vocab['w2i']

    def __len__(self):
        return len(self._i2w)

    def i2w(self, i):
        return self._i2w[i]

    def w2i(self, w):
        return int(self._w2i[w])

    def extend(self, vocab, copy=False):
        '''
        Extends the vocabulary dynamically by additional tokens
        found in 'vocab'. useful when copy attention is employed,
        where the target vocabulary is extended by the input tokens.

        :param vocab:   the vocab to extend this vocab by.
        :returns:       the extended vocab.
        '''

        if copy:
            extended_vocab = {
                'i2w': self._i2w.copy(),
                'w2i': self._w2i.copy()
            }

        else:
            extended_vocab = {
                'i2w': self._i2w,
                'w2i': self._w2i
            }

        base_i2w_len = len(self._i2w)
        base_w2i_len = len(self._w2i)

        extended_vocab['i2w'].update({
            base_i2w_len+k: v
            for k, v in vocab._i2w.items()
        })

        extended_vocab['w2i'].update({
            k: base_w2i_len+int(v)
            for k, v in vocab._w2i.items()
        })

        return Vocab(extended_vocab)

    def remove(self, vocab):
        '''
        Removes the elements in 'vocab' from this vocab. Useful
        when using copy attention to remove the sample input tokens
        a target vocabulary was extended by temporarily.

        :param vocab:   the vocabulary elements to remove from this
                        vocabulary.
        '''

        for key in vocab._w2i.keys():
            del self._w2i[key]

        ext_start = len(self._i2w) - len(vocab)
        for key in range(ext_start, len(self._i2w)):
            del self._i2w[key]
