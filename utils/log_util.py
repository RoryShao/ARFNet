import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  #

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  #
        self.logger.setLevel(self.level_relations.get(level))  #
        sh = logging.StreamHandler()  #
        sh.setFormatter(format_str)  #
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8') 
        th.setFormatter(format_str)  #
        self.logger.addHandler(sh)  #
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('info.log', level='info')
    info = 'Namespace(batch_size=128)'
    for i in range(20):
        log.logger.info("epoch "+str(i)+": "+info)