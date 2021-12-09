import os, sys
sys.path.append(os.getcwd())
cwd = os.getcwd()
sys.path.append(cwd)
import argparse

from src.yaml_utils.yaml_parser import load_and_apply_yaml_config
from NAS.ace_v2_supernet.block_decoder import ACEBlockDecoder
from NAS.ace_v2_supernet.model_builder import build_acenet_model
from random import choice

def main(args):
    # Import YAML configuration.
    yaml_config = load_and_apply_yaml_config(args.yaml_cfg)
    proto_dir = os.path.join(cwd,"proto_dir")
    if not os.path.exists(proto_dir):
        os.makedirs(proto_dir)

    acenet_block_decoder = ACEBlockDecoder()

    depth_space = [[1,2],
                   [2, 3, 4, 5],
                   [2, 3, 4, 5],
                   [4, 5, 6, 7],
                   [2, 3, 4, 5],
                   [1]]

    activation_space = [0, 5]

    for k in range(int(args.num_subnets / 2)):

        acenet_block_args = ['b0_r2_s1_f16_se0.50_a1',
                             'b1_r5_s2_f24_se0.50_a1',
                             'b2_r5_s2_f40_se0.50_a1',
                             'b3_r7_s2_f80_se0.50_a1',
                             'b4_r5_s2_f160_se0.50_a1',
                             'b4_r1_s1_f160_se0.50_a1']
        acenet_block_args_depth_change = acenet_block_args
        for i in range(len(acenet_block_args)):
            depth = choice(depth_space[i])
            block_cfg_list_depth = list(acenet_block_args[i])
            block_cfg_list_depth[4] = str(depth)
            acenet_block_args_depth_change[i] = ''.join(block_cfg_list_depth)
        model_argscope = acenet_block_decoder.decode_to_argscope(acenet_block_args_depth_change)
        #print("Decoded Argscope: ", model_argscope)
        model_name = os.path.join(proto_dir, "subnet-with-depth-changing-%d" %k)
        # Build ACENet models using default argscope. For cell-based ACENet search, please set 'no_skip_first' to True.
        proto = build_acenet_model(model_name,
                                   model_argscope=model_argscope,
                                   depth_multiplier=args.depth_multiplier,
                                   width_multiplier=args.width_multiplier,
                                   fixed_head=args.fixed_head,
                                   smart_se=True)
        prototxt = os.path.join(proto_dir, "subnet-with-depth-changing-%d.prototxt" %k)
        proto.set_pretrain_path("/home/public/acenet_supernet")
        with open(prototxt, 'w') as fp:
            proto.dump(fp)

        acenet_block_args = ['b0_r2_s1_f16_se0.50_a1',
                             'b1_r5_s2_f24_se0.50_a1',
                             'b2_r5_s2_f40_se0.50_a1',
                             'b3_r7_s2_f80_se0.50_a1',
                             'b4_r5_s2_f160_se0.50_a1',
                             'b4_r1_s1_f320_se0.50_a1']
        acenet_block_args_act_change = acenet_block_args
        for j in range(len(acenet_block_args)):
            act = choice(activation_space)
            block_cfg_list_act = list(acenet_block_args[j])
            if (len(block_cfg_list_act) == 23):
                block_cfg_list_act[18] = str(act)
            if (len(block_cfg_list_act) == 22):
                block_cfg_list_act[17] = str(act)
            acenet_block_args_act_change[j] = ''.join(block_cfg_list_act)
        model_argscope = acenet_block_decoder.decode_to_argscope(acenet_block_args_act_change)
        # print("Decoded Argscope: ", model_argscope)
        model_name = os.path.join(proto_dir, "subnet-with-act-changing-%d" % k)
        # Build ACENet models using default argscope. For cell-based ACENet search, please set 'no_skip_first' to True.
        proto = build_acenet_model(model_name,
                                   model_argscope=model_argscope,
                                   depth_multiplier=args.depth_multiplier,
                                   width_multiplier=args.width_multiplier,
                                   fixed_head=args.fixed_head,
                                   smart_se=True)
        prototxt = os.path.join(proto_dir, "subnet-with-act-changing-%d.prototxt" % k)
        proto.set_pretrain_path("/home/public/acenet_supernet")
        with open(prototxt, 'w') as fp:
            proto.dump(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_cfg", type=str, nargs='?', help="YAML config for Backend settings.",
                        default=None)
    parser.add_argument("--depth_multiplier", type=float, default=1.0,
                        help="Multiplier for depth of each stage.")
    parser.add_argument("--width_multiplier", type=float, default=1.0,
                        help="Multiplier for channel width in each stage.")
    parser.add_argument("--fixed_head", action="store_true", default=False,
                        help="Whether fix head or not")
    parser.add_argument("--proto_name", type=str, default=None,
                        help="Prototxt name for saving")
    parser.add_argument("--num_subnets", type=int, default=10,
                        help="number of subnets, even number")
    args = parser.parse_args()
    main(args)