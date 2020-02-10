#!/usr/bin/env python

import pickle
import random

nsample = 10000
in_dir = '../31_fitness/'
out_dir = 'trajs/'

def main():
    gt_graph = pickle.load(open(f'{in_dir}gt_graph.pkl', 'rb'))
    gt_mut_list = pickle.load(open(f'{in_dir}gt_mut_list.pkl', 'rb'))
    traj_list = []
    for i in range(nsample):
        v = 0
        path = [v, ]
        next_steps = step_forward(v, gt_graph, gt_mut_list)
        for __ in range(10):
            v_next = random.choice(next_steps)
            v = v_next
            path.append(v)
            next_steps = step_forward(v, gt_graph, gt_mut_list)
        traj_list.append(path)
        print(f'{i}-th trajectory found, length = {len(path)}')
    with open(f'{out_dir}02_traj_mafree.pkl', 'wb') as f:
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