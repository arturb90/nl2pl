'''
Add components to the logger in the order they should appear.
Then, invoke start to begin logging.
'''

import os

from abc import ABC, abstractmethod
from tqdm import tqdm


class Logger:
    '''
    An logger that displays information on the console and prints
    it to a file. Uses updatable components based on tqdm.

    :ivar header:       Sort of a "title" for this logger. Printed
                        before any updatable component.
    :ivar out_path:     the path of the file to write logging
                        information to.
    :ivar length:       the total length (in terms of number of lines
                        printed to the console) of this logger's updatable
                        components.
    :ivar active:       flag indicating whether this logger is currently
                        active.
    '''

    def __init__(self, header=None, out_path=None):
        self.header = header
        self.out_path = out_path
        self.length = 0
        self.active = False

        self.__lines = []
        self.__active = []
        self.__out_file = None

        self.bar_format = ('{l_bar}{bar}|'
                           '{n_fmt:>6}/{total_fmt:>6} '
                           '[{elapsed:>8}<{remaining:>8}, '
                           '{rate_fmt:>11}{postfix}]')

    def start(self):
        '''
        Initializes the declared components.
        '''

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
                    # Create a tqdm object for each line
                    # in the scroll component and add it
                    # to the list of active components.
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
            # If out file was specified, create it.
            self.__out_file = open(self.out_path, 'w+')

        self.active = True

    def log(self, text):
        # Print to output file and console. Make sure all
        # updatabale components are inactive when logging.

        if self.__out_file:
            self.__out_file.write(text + os.linesep)
            self.__out_file.flush()
        print(text)

    def update(self, pos, value):
        '''
        Update a specific position in the list of updatable
        components with 'value'.

        :param pos:     the position to update.
        :param value:   the value to update the line
                        at 'position' with.
        '''

        lines = self.__active

        for i in range(len(self.__active)):

            if pos == i:
                if lines[i].invoke_completed:
                    # If line is a bar and is completed,
                    # perform action.
                    lines[i].invoke_completed = False
                    lines[i].on_complete(lines[i])
                    lines[i].counter = value

                lines[i].ref.update(value)
                lines[i].ref.refresh()

                if lines[i].counter >= lines[i].total:
                    # If counter is at maximum capacity of
                    # bar, invoke complete action at next update.
                    lines[i].invoke_completed = True

            elif type(lines[i]) == _Bar:
                lines[i].ref.update(0)

    def close(self):
        '''
        Closes the logger, all active tqdm objects and
        the output file.
        '''

        for line in self.__active:
            line.ref.close()

        if self.__out_file:
            self.__out_file.close()

        self.active = False

    def add_bar(self, total, desc, on_complete=(lambda i: None)):
        '''
        Add a progress bar to this logger.

        :param total:       the total 'capacity' of this progress bar.
        :param desc:        this progressbar's decription.
        :param on_complete: the action to invoke on completion of
                            this progress bar.
        '''

        assert type(total) == int
        component = _Bar()

        line = {
            'format': 'bar',
            'on_complete': on_complete,
            'ref': component,
            'total': total,
            'desc': desc
        }

        # Add this component as a line to
        # the logger.
        self.__lines.append(line)
        return component

    def add_text(self, text):
        '''
        Adds a simple updatable text to the logger.

        :param text:    the inital text to display.
        '''

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
        '''
        Adds a scroll component to the logger.

        :param length:  the length of the text cache.
        '''

        component = _Scroll()

        line = {
            'format': 'scroll',
            'ref': component,
            'length': length
        }

        self.__lines.append(line)
        return component


class Component(ABC):
    '''
    The base class for each updatable component in the
    logger.

    :ivar root:     the logger associated with this
                    component.
    :ivar type:     the type of this component as string.
    :ivar ref:      the tqdm object associated with this component.
    :ivar pos:      the (starting) position of this component
                    in terms of the line relative to the line
                    of the first updatable component in the logger.
    '''

    def __init__(self):
        self.root = None
        self.type = None
        self.ref = None
        self.pos = None

    @abstractmethod
    def update():
        NotImplemented


class _Bar(Component):
    '''
    Progress bar component.

    :ivar on_complete:      action to invoke when bar is full.
    :ivar invoke_completed: flag to set when 'on_complete' should
                            be invoked.
    :ivar type:             type of this component as string.
    :ivar counter:          current integer bar progress.
    :ivar total:            total integer bar "capacity".
    '''

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
    '''
    A simple, single line of updatable text component.

    :ivar type:     type of this component as string.
    '''

    def __init__(self):
        super(_Text, self).__init__()
        self.type = 'text'

    def update(self, text):
        assert type(text) == str
        self.ref.set_description_str(text)

    def close(self):
        self.ref.close()


class _Scroll(Component):
    '''
    A text component that keeps a cache of length 'length'
    and displays all texts in the cache. Whenever a line of
    text is added to this component it appears at the bottom
    and pushes the preceding messages up, and the oldest message
    is pushed out of the cache.

    :ivar type:     type of this component as string.
    :ivar length:   length of the cache.
    :ivar cache:    cache holding the 'length' latest
                    messages.
    :ivar ref:      list that holds references to the
                    tqdm objects used in this component.
    '''

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
            # Update all texts to reflect the
            # current sequence of texts in the cache.
            text = self.cache[i]
            line = self.ref[self.length-i-1]
            line.set_description_str(text)

        if len(self.cache) > self.length:
            # Remove oldest message from cache.
            remove = len(self.cache)-self.length
            del self.cache[-remove:]
