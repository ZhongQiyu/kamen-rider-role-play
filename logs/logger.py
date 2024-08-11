# logger.py

import logging
import os

class LoggerManager:
    def __init__(self, log_directory='logs'):
        # 确保日志目录存在
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # 创建logger
        self.logger = logging.getLogger('kamen_rider_blade')
        self.logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器和级别设置
        levels = {'app.log': logging.INFO, 'error.log': logging.ERROR, 'debug.log': logging.DEBUG}
        for filename, level in levels.items():
            handler = logging.FileHandler(os.path.join(log_directory, filename))
            handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 创建一个单独的错误日志记录器
        self.error_logger = logging.getLogger('error_logger')
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(os.path.join(log_directory, 'error.log'))
        error_handler.setFormatter(formatter)  # 使用之前定义的格式器
        self.error_logger.addHandler(error_handler)

    def get_logger(self):
        return self.logger

    def get_error_logger(self):
        return self.error_logger
