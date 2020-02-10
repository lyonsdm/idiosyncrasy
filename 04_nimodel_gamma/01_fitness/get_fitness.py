#!/usr/bin/env python

import itertools
import numpy
import os
import pickle
import subprocess

N = 16
out_dir = f'./'
states = 'AT'


def main():
    numpy.random.seed(4032)
    gt_list = [''.join(x) for x in itertools.product(states, repeat=N)]
    icombn_list = [tuple(itertools.combinations(list(range(N)), x + 1))
                   for x in range(N)]
    idict_list = []
    for k in range(N):
        icombn_cnt = len(icombn_list[k])
        state_combns = [
            ''.join(x) for x in itertools.product(states, repeat=k + 1)
        ]
        print(f'{k}-order I, {icombn_cnt} site combinations,'
              f' {len(state_combns)} state combinations')
        idicts = []
        for j in range(icombn_cnt):
            ivals = numpy.random.gamma(shape=1, size=len(state_combns))
            idicts.append(dict(zip(state_combns, ivals)))
        idict_list.append(dict(zip(icombn_list[k], idicts)))
    print('='*50)
    with open(f'{out_dir}idict_list.pkl', 'wb') as f:
        pickle.dump(idict_list, file=f)
    curr_logfit_list = [0 for __ in gt_list]
    for nlevel in range(N):
        print(nlevel, flush=True)
        dir_name = f'{out_dir}nlevel_{nlevel:02d}/'
        if not os.path.isdir(dir_name):
            subprocess.call(['mkdir', dir_name])
        gt_logfit_list = []
        for i, gt in enumerate(gt_list):
            if i % 1000 == 0:
                print(i, end=' ', flush=True)
            logfit = get_logfit(gt, idict_list, nlevel, curr_logfit_list[i])
            gt_logfit_list.append(logfit)
            curr_logfit_list[i] = logfit
        print(flush=True)
        gt_dict = dict(zip(gt_list, list(range(len(gt_list)))))
        with open(f'{dir_name}gt_mut_list.pkl', 'wb') as f:
            pickle.dump(gt_list, file=f)
        with open(f'{dir_name}gt_logfit_raw.pkl', 'wb') as f:
            pickle.dump(gt_logfit_list, file=f)
        with open(f'{dir_name}gt_dict.pkl', 'wb') as f:
            pickle.dump(gt_dict, file=f)


def get_logfit(gt, idict_list, nlevel, curr_logfit):
    # print(gt)
    logfit = curr_logfit
    for icombn in list(idict_list[nlevel].keys()):
        gt_states = ''.join([gt[x] for x in icombn])
        logfit += idict_list[nlevel][icombn][gt_states]
    return logfit



if __name__ == '__main__':
    main()
