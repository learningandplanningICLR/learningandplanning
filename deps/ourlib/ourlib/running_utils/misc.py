


# COPYRIGHT: Tee classes copied from https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
import sys


class TeeStdout(object):
    def __init__(self, path, mode):
        self.file = open(path, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class TeeStderr(object):
    def __init__(self, path, mode):
        self.file = open(path, mode)
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()
