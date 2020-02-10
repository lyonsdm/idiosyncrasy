#!/usr/bin/env python

import pickle
import random

import numpy

nsample = 10000
ntry = 10
path_minlength  = 3
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
        floor = gt_fit_arr.argmin()
        traj_list = []
        for i in range(nsample):
            v = floor
            path = [v, ]
            v_next = step_forward(v, gt_graph, gt_mut_list, gt_fit_arr)
            while v_next != None:
                v = v_next
                path.append(v)
                v_next = step_forward(v, gt_graph, gt_mut_list, gt_fit_arr)
            if len(path) >= path_minlength:
                traj_list.append(path)
                print(f'{len(traj_list)}-th trajectory '
                      f'found, length = {len(path)}')
        with open(f'{out_dir}03_traj_adapt_{level:02d}.pkl', 'wb') as f:
            pickle.dump(traj_list, f)
        print(f'total {len(traj_list)} trajectories')


def step_forward(v, gt_graph, gt_mut_list, gt_fit_list):
    up_list = []
    v1_list = [x[0] for x in gt_graph[v]]
    for v1 in v1_list:
        if gt_fit_list[v1] > gt_fit_list[v]:
            up_list.append(v1)
    if len(up_list) > 0:
        return random.sample(up_list, 1)[0]
    return None


if __name__ == '__main__':
    main()