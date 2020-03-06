import sys
import pprint
import logging
import datetime

class Logger():
    def __init__(self, save_log=True, log_path=""):


        currdate=str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))
        log = logging.getLogger(currdate)
        log.setLevel(level=logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(levelname)s:%(asctime)s; %(message)s","%Y-%m-%d %H:%M:%S")

        if save_log:
            fh = logging.FileHandler(log_path + currdate + ".log")
            fh.setFormatter(formatter)

        # reate console handler for logger.
        ch = logging.StreamHandler()
        ch.setLevel(level=logging.DEBUG)
        ch.setFormatter(formatter)

        # add handlers to logger.
        if save_log:
            log.addHandler(fh)

        log.addHandler(ch)

        self.log = log 

    def info(self, message, pp=False):
        if pp:
            message=pprint.pformat(message)
        self.log.info(message)
    
    def exception(self, message):
        self.log.exception(message)