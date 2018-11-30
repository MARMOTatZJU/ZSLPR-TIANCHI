# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：useful wheels
# Author: Yinda XU

import os
from contextlib import contextmanager
import time
@contextmanager
def print_elapsed_time(prompt='this part'):
    time_start = time.time()
    yield
    print('Elapsed time of {:s} : {:.2f}[s]'.format(str(prompt), time.time()-time_start))


# Get current time for TimeStemp
def getTimeStamp():
    actTime = time.localtime()
    TimeStamp = '{YYYY:04d}{MM:02d}{DD:02d}-{H:02d}H{M:02d}M{S:02d}'.format(
        YYYY=actTime.tm_year, MM=actTime.tm_mon, DD=actTime.tm_mday,
        H=actTime.tm_hour, M=actTime.tm_min, S=actTime.tm_sec,
    )
    return TimeStamp

class AverageMeter(object):
    """Computes and stores the average and current value, copied from Pytorch Example"""
    def __init__(self, NVars=1, ):
        self.NVars=NVars
        self.reset()

    def reset(self, ):
        self.sum = [0 for i in range(self.NVars)]
        self.cnt = 0

    def update(self, val, n=1):
        for i in range(self.NVars):
            self.sum[i] += val[i] * n
        self.cnt += n
    def avg(self,):
        if self.cnt:
            return tuple([self.sum[i]/self.cnt for i in range(self.NVars)])
        else:
            return None
