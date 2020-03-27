import logging
from typing import List, Tuple

import colorlog


class RLogger(logging.Logger):

    def __init__(self,
                 name: str = 'pandemonium',
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

        # Set up a FileHandler
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(name)-5s: [%(levelname)-8s] [%(processName)-10s] '
            '[%(threadName)-9s] [%(asctime)s] %(message_log_color)s%(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            log_colors={
                'DEBUG': 'black',
                'INFO': 'black',
                'WARNING': 'black',
                'ERROR': 'black',
                'CRITICAL': 'black',
            },
            secondary_log_colors={
                'message': {
                    'DEBUG': 'white',
                    'INFO': 'cyan',
                    'WARNING': 'yellow',
                    'ERROR': 'purple',
                    'CRITICAL': 'bold_red',
                }
            }
        )

        if file_logger is not None:
            self.file_handler = logging.FileHandler(file_logger[1], mode='a')
            self.file_handler.setLevel(file_logger[0])
            self.file_handler.setFormatter(formatter)
            self.addHandler(self.file_handler)

        # Set up a StreamHandler
        if level is not None:
            self.console = logging.StreamHandler()
            self.console.setLevel(level)
            self.console.setFormatter(formatter)
            self.addHandler(self.console)

        if extra_handlers is not None:
            for hdlr in extra_handlers:
                self.addHandler(hdlr)
