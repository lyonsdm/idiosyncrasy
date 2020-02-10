#!/usr/bin/env python

import pickle

import numpy

for level in range(16):
    raw_logfit = numpy.array(pickle.load(open(
        f'nlevel_{level:02d}/gt_logfit_raw.pkl', 'rb'
    )))
    logfit = (raw_logfit - raw_logfit.min())/(raw_logfit.max() - raw_logfit.min())
    with open(f'nlevel_{level:02d}/gt_logfit_list.pkl', 'wb') as f:
        pickle.dump(logfit, file=f)