import logging
from typing import List, Tuple, Union

import colorlog
from termcolor import colored

Key = Union[int, str]
Value = Union[bytes, str, float]


class RLogger(logging.Logger):

    def __init__(self,
                 name: str = 'RLogger',
                 level: int = logging.NOTSET,
                 file_logger: Tuple[int, str] = None,
                 extra_handlers: List[logging.Handler] = None,
                 ):
        """ Basic logger that can write to a file on disk or to stderr.

        :param name: logger reference name
        :param level: logging verbosity level
        :param file_logger: name of the file to log to along with the loglevel
        """
        super().__init__(name=name, level=level)

        # Set up main formatter
        formatter = colorlog.ColoredFormatter(
            f'%(log_color)s %(name)-{len(name)}s | %(levelname)-8s | '
            f'%(processName)-12s | %(threadName)-12s | [%(asctime)s] '
            f'%(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            log_colors={
                'DEBUG': 'white',
                'INFO': 'white',
                'WARNING': 'white',
                'ERROR': 'white',
                'CRITICAL': 'white'}
        )

        # Set up file handler
        if file_logger is not None:
            file_handler = logging.FileHandler(file_logger[1], mode='a')
            file_handler.setLevel(file_logger[0])
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

        # Set up a StreamHandler
        if level is not None:
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            self.addHandler(console)

        if extra_handlers is not None:
            for hdlr in extra_handlers:
                self.addHandler(hdlr)

    def debug(self, msg, color='white', attrs=None, *args, **kwargs):
        super().debug(colored(msg, color, attrs=attrs), *args, **kwargs)

    def unhandled(self, msg, color='blue', attrs=None, *args, **kwargs):
        super().debug(colored(msg, color, attrs=attrs), *args, **kwargs)

    def info(self, msg, color='green', attrs=None, *args, **kwargs):
        super().info(colored(msg, color, attrs=attrs, *args, **kwargs))

    def warning(self, msg, color='blue', attrs=None, *args, **kwargs):
        super().warning(colored(msg, color, attrs=attrs), *args, **kwargs)

    def error(self, msg, color='magenta', attrs=None, *args, **kwargs):
        super().error(colored(msg, color, attrs=attrs), *args, **kwargs)

    def exception(self, msg, color='magenta', attrs=None, *args, **kwargs):
        """ Creates a log message similar to ``ProjectLogger.error``.

        The difference is that `exception` dumps a stack trace along with it.
        Call this method only from an exception handler.
        """
        super().exception(colored(msg, color, attrs=attrs), *args, **kwargs)

    def critical(self, msg, color='grey', on_color='on_red', attrs=None, *args,
                 **kwargs):
        super().critical(colored(msg, color, on_color, attrs), *args, **kwargs)


def test_logger():
    logger = RLogger(level=10,
                     file_logger=(10, 'test.log'),
                     extra_handlers=[])
    logger.debug('Nevermind me.')
    logger.info('1010010010010011010')
    logger.warning('Hey there!')
    logger.error('¯\\_(ツ)_/¯')
    try:
        x = 0 / 0
    except ZeroDivisionError:
        logger.exception('To infinity and beyond!')
    logger.critical('ALERT ALERT ALERT', attrs=['blink'])


if __name__ == '__main__':
    test_logger()
