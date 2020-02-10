#!/usr/bin/env python

import pickle
import random

import numpy

nsample = 10000
ntry = 5
path_minlength  = 3
in_dir = '../31_fitness/'
out_dir = 'trajs/'

def main():
    gt_graph = pickle.load(open(f'{in_dir}gt_graph.pkl', 'rb'))
    gt_mut_list = pickle.load(open(f'{in_dir}gt_mut_list.pkl', 'rb'))
    gt_fit_list = pickle.load(open(f'{in_dir}gt_fit_list.pkl', 'rb'))
    floor_list = numpy.where(numpy.array(gt_fit_list) <= 0.5)[0]
    # floor_sampled = numpy.random.choice(floor_list, size=nsample, replace=False)
    floor_sampled = floor_list
    traj_list = []
    for i, v in enumerate(floor_sampled):
        for __ in range(ntry):
            path = [v, ]
            v_next = step_forward(v, gt_graph, gt_mut_list, gt_fit_list)
            while v_next != None:
                v = v_next
                path.append(v)
                v_next = step_forward(v, gt_graph, gt_mut_list, gt_fit_list)
            if len(path) >= path_minlength:
                traj_list.append(path)
                print(f'{len(traj_list)}-th trajectory '
                      f'found, length = {len(path)}')
                continue
        # traj_list.append(path)
    with open(f'{out_dir}03_traj_adapt.pkl', 'wb') as f:
        pickle.dump(traj_list, f)
    print(f'total {len(traj_list)} trajectories')


def step_forward(v, gt_graph, gt_mut_list, gt_fit_list):
    up_list = []
    v1_list = [x[0] for x in gt_graph[v]]
    for v1 in v1_list:
        if gt_fit_list[v1] > gt_fit_list[v]:
            up_list.append(v1)
        # else:
        #     fit_ratio = gt_fit_list[v1] / gt_fit_list[v]
        #     if numpy.random.random() < fit_ratio:
        #         up_list.append(v1)
    if len(up_list) > 0:
        return random.sample(up_list, 1)[0]
    return None


if __name__ == '__main__':
    main()