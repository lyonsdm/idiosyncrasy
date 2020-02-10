#!/usr/bin/env python

import pickle
import random

import numpy

nsample = 10000
in_dir_all = '../01_fitness/'
out_dir = 'trajs/'

def main():
    for level in numpy.arange(16):
        in_dir = f'{in_dir_all}nlevel_{level:02d}/'
        gt_graph = pickle.load(open(f'{in_dir}gt_graph.pkl', 'rb'))
        gt_mut_list = pickle.load(open(f'{in_dir}gt_mut_list.pkl', 'rb'))
        gt_fit_arr = numpy.array(pickle.load(
            open(f'{in_dir}gt_logfit_list.pkl', 'rb')
        ))
        wt_fit = gt_fit_arr.min() + 0.9 * (gt_fit_arr.max() - gt_fit_arr.min())
        wt_idx = numpy.absolute(gt_fit_arr - wt_fit).argmin()
        traj_list = []
        for i in range(nsample):
            v = wt_idx
            path = [v, ]
            next_steps = step_forward(v, gt_graph, gt_mut_list)
            for __ in range(10):
                v_next = random.choice(next_steps)
                v = v_next
                path.append(v)
                next_steps = step_forward(v, gt_graph, gt_mut_list)
            traj_list.append(path)
            print(f'{i}-th trajectory found, length = {len(path)}')
        with open(f'{out_dir}02_traj_mafree_{level:02d}.pkl',
                  'wb') as f:
            pickle.dump(traj_list, f)


def step_forward(v, gt_graph, gt_mut_list):
    # res_list = []
    v1_list = [x[0] for x in gt_graph[v]]
    # for v1 in v1_list:
    #     if len(gt_mut_list[v1]) - len(gt_mut_list[v]) == 1:
    # res_list.append(v1)
    return v1_list


if __name__ == '__main__':
    main()