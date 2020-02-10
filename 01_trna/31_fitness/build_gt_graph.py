#!/usr/bin/env python

import pickle
import time

fit_data_file = '../00_data/ExpData.txt'
in_dir = './'
out_dir = './'
wildtype = open(fit_data_file).readlines()[1].split()[3]


def main():
    gt_mut_list = pickle.load(open(f'{in_dir}/gt_mut_list.pkl', 'rb'))
    gt_graph = []
    for j in range(len(gt_mut_list)):
        neighbor_list = []
        cnt = 0
        for i in range(len(gt_graph)):
            diff = list(gt_mut_list[j] ^ gt_mut_list[i])
            if (len(diff) == 1) or (len(diff) == 2 and diff[0][0] == diff[1][0]):
                pos = diff[0][0]
                start = get_nt(gt_mut_list[j], pos)
                end = get_nt(gt_mut_list[i], pos)
                neighbor_list.append((i, (pos, start, end)))
                gt_graph[i].append((j, (pos, end, start)))
                cnt += 1
        gt_graph.append(neighbor_list)
        print(f'{j}-th genotype added to graph, found {cnt} neighbors')
        print(time.asctime(time.localtime()))
    with open(f'{out_dir}/gt_graph.pkl', 'wb') as f:
        pickle.dump(gt_graph, file=f)


def get_nt(gt, pos):
    muts = list(gt)
    mut_pos = [x[0] for x in muts]
    if pos in mut_pos:
        return muts[mut_pos.index(pos)][1]
    else:
        return wildtype[pos]


if __name__ == '__main__':
    main()