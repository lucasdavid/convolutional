import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def get_time_hhmmss(self):
        m, s = divmod(self.elapsed(), 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str
