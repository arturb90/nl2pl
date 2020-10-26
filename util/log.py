import os

from abc import ABC, abstractmethod
from tqdm import tqdm


class Logger:

    def __init__(self, header=None, out_path=None):
        self.header = header
        self.out_path = out_path
        self.length = 0
        self.active = False

        self.__lines = []
        self.__active = []
        self.__out_file = None

        self.bar_format = ('{l_bar}{bar}|' +
                           '{n_fmt:>6}/{total_fmt:>6} ' +
                           '[{elapsed:>8}<{remaining:>8}, ' +
                           '{rate_fmt:>11}{postfix}]')

    def start(self):
        if self.header:
            print(self.header)

        for line in self.__lines:

            if line['format'] == 'text':
                item = tqdm(
                    total=0,
                    desc=line['text'],
                    bar_format='{desc}',
                    position=self.length
                )

                line['ref'].root = self
                line['ref'].ref = item
                line['ref'].pos = self.length

                self.__active.append(line['ref'])
                self.length += 1

            elif line['format'] == 'bar':

                item = tqdm(
                    total=line['total'],
                    desc=line['desc'],
                    bar_format=self.bar_format,
                    position=self.length
                )

                line['ref'].root = self
                line['ref'].ref = item
                line['ref'].pos = self.length
                line['ref'].total = line['total']
                line['ref'].on_complete = line['on_complete']

                self.__active.append(line['ref'])
                self.length += 1

            elif line['format'] == 'scroll':
                line['ref'].pos = self.length
                line['ref'].length = line['length']

                for i in range(line['length']):
                    item = tqdm(
                        total=0,
                        desc='',
                        bar_format='{desc}',
                        position=self.length
                    )

                    line['ref'].ref.append(item)

                    text = _Text()
                    text.root = line['ref'].ref
                    text.pos = self.length
                    text.ref = item

                    self.__active.append(text)
                    self.length += 1

        if self.out_path:
            self.__out_file = open(self.out_path, 'w+')
        
        self.active = True

    def log(self, text):
        if self.__out_file:
            self.__out_file.write(text + os.linesep)
            self.__out_file.flush()
        print(text)

    def update(self, pos, value):
        lines = self.__active

        for i in range(len(self.__active)):
            if pos == i:
                if lines[i].invoke_completed:
                    lines[i].invoke_completed = False
                    lines[i].on_complete(lines[i])
                    lines[i].counter = value

                lines[i].ref.update(value)
                lines[i].ref.refresh()

                if lines[i].counter >= lines[i].total:
                    lines[i].invoke_completed = True

            elif type(lines[i]) == _Bar:
                lines[i].ref.update(0)

    def close(self):
        for line in self.__active:
            line.ref.close()

        if self.__out_file:
            self.__out_file.close()

        self.active = False

    def add_bar(self, total, desc, on_complete=(lambda i: None)):
        assert type(total) == int
        component = _Bar()

        line = {
            'format': 'bar',
            'on_complete': on_complete,
            'ref': component,
            'total': total,
            'desc': desc
        }

        self.__lines.append(line)
        return component

    def add_text(self, text):
        assert type(text) == str
        component = _Text()

        line = {
            'format': 'text',
            'ref': component,
            'text': text
        }

        self.__lines.append(line)
        return component

    def add_scroll(self, length):
        component = _Scroll()

        line = {
            'format': 'scroll',
            'ref': component,
            'length': length
        }

        self.__lines.append(line)
        return component


class Component(ABC):

    def __init__(self):
        self.root = None
        self.type = None
        self.ref = None
        self.pos = None

    @abstractmethod
    def update():
        NotImplemented


class _Bar(Component):

    def __init__(self):
        super(_Bar, self).__init__()
        self.on_complete = (lambda i: None)
        self.invoke_completed = False
        self.type = 'bar'
        self.counter = 0
        self.total = 0

    def update(self, value):
        assert type(value) == int
        self.counter = self.counter + value
        self.root.update(self.pos, value)


class _Text(Component):

    def __init__(self):
        super(_Text, self).__init__()
        self.type = 'text'

    def update(self, text):
        assert type(text) == str
        self.ref.set_description_str(text)

    def close(self):
        self.ref.close()


class _Scroll(Component):

    def __init__(self):
        super(_Scroll, self).__init__()
        self.type = 'scroll'
        self.length = 1
        self.cache = []
        self.ref = []

    def update(self, text):
        assert type(text) == str
        self.cache.insert(0, text)
        len_ = min(len(self.cache), self.length)

        for i in reversed(range(len_)):
            text = self.cache[i]
            line = self.ref[self.length-i-1]
            line.set_description_str(text)

        if len(self.cache) > self.length:
            remove = len(self.cache)-self.length
            del self.cache[-remove:]
