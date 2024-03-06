import logging


def get_logger(name='blueberry', file_path='logger.log', level=logging.DEBUG):
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(file_path)
    fh.setLevel(level)

    # 再创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger