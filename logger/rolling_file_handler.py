import sys
from logging.handlers import RotatingFileHandler
import os

class RollingFileHandler(RotatingFileHandler):

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super(RollingFileHandler, self).__init__(filename=filename,
                                                 mode=mode,
                                                 maxBytes=maxBytes,
                                                 backupCount=backupCount,
                                                 encoding=encoding,
                                                 delay=delay)

        self.last_backup_cnt = self.find_last_backup_cnt()

    def find_last_backup_cnt(self):
        # 尋找目前log檔中的最大值
        for i in range(1, sys.maxsize):
            nextName = f"{self.baseFilename}.{i}"
            if not os.path.exists(nextName):
                # 要回傳找不到檔案的前一個編號
                return i - 1


    # override
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # my code starts here
        self.last_backup_cnt += 1
        nextName = "%s.%d" % (self.baseFilename, self.last_backup_cnt)
        self.rotate(self.baseFilename, nextName)
        # my code ends here
        if not self.delay:
            self.stream = self._open()