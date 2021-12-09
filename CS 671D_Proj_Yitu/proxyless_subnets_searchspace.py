import random
def proxyless_search_space():
    # in_chs, out_chs, strides, k_size, exp_factor, #replicates, activation, se_ratio
    stage0_cfg = [[32],  # in_chs
                  [16],  # out_chs
                  [1],  # strides
                  [3, 5, 7],  # k_size
                  [1],  # exp_factor
                  [1],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage1_cfg = [[], # in_chs
                  [16, 24, 32],  # out_chs
                  [2],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [1, 2, 3],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage2_cfg = [[],  # in_chs
                  [24, 40, 64],  # out_chs
                  [2],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [1, 2, 3],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage3_cfg = [[],  # in_chs
                  [64, 80, 96],  # out_chs
                  [2],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [2, 3, 4],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage4_cfg = [[],  # in_chs
                  [96, 112, 128],  # out_chs
                  [1],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [2, 3, 4],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage5_cfg = [[],  # in_chs
                  [160, 192, 224],  # out_chs
                  [2],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [3, 4, 5],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    stage6_cfg = [[],  # in_chs
                  [320],  # out_chs
                  [1],  # strides
                  [3, 5, 7],  # k_size
                  [3, 6],  # exp_factor
                  [1],  # replicates
                  ['swish'],  # activation
                  [0]]  # se_ratio

    search_space_list = [stage0_cfg, stage1_cfg, stage2_cfg, stage3_cfg, stage4_cfg, stage5_cfg, stage6_cfg]

    return  search_space_list

def change_ksize_width(search_space_list):
    # in_chs, out_chs, strides, k_size, exp_factor, #replicates, activation, se_ratio
    ops_def = [(32, 16, 1, 7, 1, 1, 'swish', 0),
               (16, 32, 2, 7, 6, 3, 'swish', 0),
               (32, 64, 2, 7, 6, 3, 'swish', 0),
               (64, 96, 2, 7, 6, 4, 'swish', 0),
               (96, 128, 1, 7, 6, 4, 'swish', 0),
               (128, 224, 2, 7, 6, 5, 'swish', 0),
               (224, 320, 1, 7, 6, 1, 'swish', 0)]

    for stageID in range(len(ops_def)):
        ops_def[stageID] = list(ops_def[stageID])
        ops_def[stageID][1] = random.choice(search_space_list[stageID][1]) # change out_chs (width)
        ops_def[stageID][3] = random.choice(search_space_list[stageID][3]) # change kernel size
        ops_def[stageID][4] = random.choice(search_space_list[stageID][4]) # change expansion factor
        if stageID >= 1:
            ops_def[stageID][0] = ops_def[stageID - 1][1]
        ops_def[stageID] = tuple(ops_def[stageID])

    return ops_def


def change_depth(search_space_list):
    # in_chs, out_chs, strides, k_size, exp_factor, #replicates, activation, se_ratio
    ops_def = [(32, 16, 1, 7, 1, 1, 'swish', 0),
               (16, 32, 2, 7, 6, 3, 'swish', 0),
               (32, 64, 2, 7, 6, 3, 'swish', 0),
               (64, 96, 2, 7, 6, 4, 'swish', 0),
               (96, 128, 1, 7, 6, 4, 'swish', 0),
               (128, 224, 2, 7, 6, 5, 'swish', 0),
               (224, 320, 1, 7, 6, 1, 'swish', 0)]
    print(ops_def)
    for stageID in range(len(ops_def)):
        ops_def[stageID] = list(ops_def[stageID])
        ops_def[stageID][5] = random.choice(search_space_list[stageID][5]) # change #layers (depth)
        ops_def[stageID] = tuple(ops_def[stageID])

    return ops_def















