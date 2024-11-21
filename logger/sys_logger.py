import sys
import logging

import matplotlib

from .rolling_file_handler import RollingFileHandler

sys_logger = logging.getLogger('sys_logger')

# 每個日誌檔案最大為 10MB，無限保留備份文件
handler = RollingFileHandler(f"./sys.stdout.log", encoding='utf-8', maxBytes=1e7)

# 設定日誌級別
handler.setLevel(logging.DEBUG)
sys_logger.setLevel(logging.DEBUG)

# 建立格式器，用於指定日誌訊息的格式
class CustomFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# 將格式器新增至handler
handler.setFormatter(CustomFormatter())

# 將handler新增至根日誌記錄器
sys_logger.addHandler(handler)

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO, stream=sys.stdout):
        self.logger = logger
        self.log_level = log_level
        self._internal_write = False
        self.stream = stream  # 保留對原始 stdout 的引用

    def write(self, buf):
        # 使用旗標避免無限遞歸
        if self._internal_write:
            return
        self._internal_write = True
        try:
            for line in buf.rstrip().splitlines():
                # 此處呼叫log時會再次調用write方法，但因為旗標已經設定為True，所以不會再次進入此分支
                self.logger.log(self.log_level, line.rstrip())
            self.stream.write(buf)  # 同時寫入原始 stdout
        finally:
            # 重置旗標
            self._internal_write = False

    def flush(self):
        self.stream.flush()  # 確保原始 stdout 的緩衝區也被刷新
    
    def isatty(self):
        return self.stream.isatty()  # 委託給原始 stdout 的 isatty 方法

# 重定向 stdout 和 stderr
sys.stdout = StreamToLogger(sys_logger, logging.INFO)
sys.stderr = StreamToLogger(sys_logger, logging.ERROR)

# 設定matplotlib的日誌級別
matplotlib.set_loglevel (level = 'warning')