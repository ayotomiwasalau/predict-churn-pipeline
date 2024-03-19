import logging


class Logger:
    def __init__(self, logname='logs/main.log'):
        """
        Initialize the Logger class.

        Args:
            logname (str, optional): The name of the log file. Defaults to 'logs/main.log'.
        """
        self.logging = logging
        self.logging.basicConfig(
            filename=logname,
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')
