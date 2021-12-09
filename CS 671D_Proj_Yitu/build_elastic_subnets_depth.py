import os, sys

cwd = os.getcwd()
sys.path.append(cwd)

import argparse
from src.engine.graph.proto.writer import ProtoWriter
from src.engine.graph.proto.layer import *
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config
from NAS.ace_v2_supernet.proxyless_nas_space.proxyless_nas_subnets.proxyless_subnets_searchspace import *
def make_divisible(n, divided_by=8):
    return divided_by * round(n / divided_by)


def add_to_scope(scope_name, name):
    return os.path.join(scope_name, name)


def replace_scope_name(scope_name, name):
    scope_names = scope_name.split("/")[:-1]
    return os.path.join(*scope_names, name)


def add_inverted_bottleneck_proto(writer, in_channel,
                                  max_kernel_def,
                                  out_channel,
                                  op_ksize,
                                  expansion, activation,
                                  strides, ifstream, ofstream,
                                  regularizer, use_se=False):
    assert ifstream is not None and ofstream is not None, 'Either ifstream or ofstream can be empty.'

    # k_size, k_in_max, k_out_max, exp_factor_max

    if expansion != 1:
        inode_name = ifstream
        onode_name = replace_scope_name(ofstream, "node_expanded")
        writer.add(ElasticConvolutional_Proto(name=onode_name,
                                              input=inode_name,
                                              filters=in_channel * expansion,
                                              kernel_size=1,
                                              batchnorm=True,
                                              strides=1,
                                              activation=activation,
                                              padding='SAME',
                                              regularizer_strength=regularizer,
                                              shared_kernel_shape=[max_kernel_def[0],
                                                                   max_kernel_def[0],
                                                                   max_kernel_def[1],
                                                                   max_kernel_def[1] * max_kernel_def[3]]))
        se_ratio = 1. / 24
    else:
        onode_name = ifstream
        se_ratio = 1. / 4
    inode_name = onode_name
    onode_name = replace_scope_name(ofstream, "node_dwact")
    writer.add(ElasticDepthwiseConv_Proto(name=onode_name,
                                          input=inode_name,
                                          kernel_size=op_ksize,
                                          batchnorm=True,
                                          strides=strides,
                                          activation=activation,
                                          padding='SAME',
                                          regularizer_strength=regularizer,
                                          shared_kernel_shape=[max_kernel_def[0],
                                                               max_kernel_def[0],
                                                               max_kernel_def[1] * max_kernel_def[3], 1]))
    if use_se == True:
        inode_name = onode_name
        onode_name = replace_scope_name(ofstream, "node_expanded_se")
        writer.add(ElasticSE_Proto(name=onode_name,
                                   se_ratio=se_ratio,
                                   input=inode_name,
                                   max_filters=max_kernel_def[1] * max_kernel_def[3]))
    inode_name = onode_name
    onode_name = replace_scope_name(ofstream, "node_projected")

    writer.add(ElasticConvolutional_Proto(name=onode_name,
                                          input=inode_name,
                                          filters=out_channel,
                                          kernel_size=1,
                                          batchnorm=True,
                                          strides=1,
                                          activation="linear",
                                          padding='SAME',
                                          regularizer_strength=regularizer,
                                          shared_kernel_shape=[max_kernel_def[0],
                                                               max_kernel_def[0],
                                                               max_kernel_def[1] * max_kernel_def[3],
                                                               max_kernel_def[2]]))

    if in_channel == out_channel and strides == 1:
        inode_name = onode_name
        onode_name = ofstream
        writer.add(Add_proto(name=onode_name,
                             input=[ifstream, inode_name],
                             activation='linear'))
    else:
        writer.add(Identity_proto(name=ofstream,
                                  input=onode_name))


def main(args):

    # Import YAML configuration.
    yaml_config = load_and_apply_yaml_config(args.yaml_cfg)
    proto_dir = os.path.join(cwd, "proto_dir_elastic_subnets_depth")
    search_space_list = proxyless_search_space()
    if not os.path.exists(proto_dir):
        os.makedirs(proto_dir)

    for id in range(args.budget):
        proto = ProtoWriter("elastic-subnet-depth-%d" % id)
        # Note: Check the stem conv number here. Should be 32 for mnasnet-a1.
        proto.add_header('imagenet')
        # Add stem for MnasNet
        proto.add(Convolutional_Proto(name="conv_pre1",
                                      input="input",
                                      kernel_size=3,
                                      strides=2,
                                      filters=args.num_stem_convs,
                                      activation=args.stem_activation_fn,
                                      regularizer_strength=args.regularizer,
                                      batchnorm=True,
                                      use_bias=False,
                                      trainable=True,
                                      use_dense_initializer=False))

        # in_chs, out_chs, strides, k_size, exp_factor, #replicates, activation, se_ratio
        ops_def = change_depth(search_space_list)

        # k_size, k_in_max, k_out_max, exp_factor_max, se_max
        max_shared_kernel_def = [
            [7, 32, 16, 1],
            [7, 16, 32, 6],
            [7, 32, 64, 6],
            [7, 64, 96, 6],
            [7, 96, 128, 6],
            [7, 128, 224, 6],
            [7, 224, 320, 6], ]

        ifstream = "conv_pre1"

        proto.add(Identity_proto(name="Stage_0/block_0/input",
                                 input=ifstream))
        ifstream = "Stage_0/block_0/input"

        for op_list_id in range(len(ops_def)):
            op = ops_def[op_list_id]
            in_chs, out_chs, strides, k_size, exp_factor, replicates, activation, se_opt = op
            se_opt = (se_opt == 0.25)
            for j in range(replicates):
                if j == 0:
                    in_channels, out_channels = in_chs, out_chs
                else:
                    in_channels, out_channels = out_chs, out_chs
                ofstream = "Stage_%d/block_%d/output" % (op_list_id, j)
                print(ofstream)
                if j == 0:
                    strides_ = strides
                    max_shared_kernel_def_ = max_shared_kernel_def[op_list_id]
                else:
                    strides_ = 1
                    max_shared_kernel_def_ = max_shared_kernel_def[op_list_id]
                    max_shared_kernel_def_[1] = max_shared_kernel_def_[2]

                add_inverted_bottleneck_proto(proto,
                                              in_channel=in_channels,
                                              out_channel=out_channels,
                                              op_ksize=k_size,
                                              expansion=exp_factor,
                                              strides=strides_,
                                              activation=activation,
                                              regularizer=args.regularizer,
                                              ifstream=ifstream,
                                              ofstream=ofstream,
                                              use_se=se_opt,
                                              max_kernel_def=max_shared_kernel_def_)
                ifstream = ofstream

        # Add final layers for MnasNet, including the Head, AvgPool etc.
        proto.add(Convolutional_Proto(name='conv1x1',
                                      input=ifstream,
                                      kernel_size=1,
                                      filters=args.num_head_convs,
                                      strides=1,
                                      activation=args.head_activation_fn,
                                      batchnorm=True,
                                      regularizer_strength=args.regularizer,
                                      use_bias=False,
                                      trainable=True,
                                      use_dense_initializer=False))
        # Add the average pool layer.
        proto.add(GlobalAvgPool_proto(name="avg_1k",
                                      input="conv1x1"))
        proto.add(Dropout_proto(name="fc_drop",
                                input="avg_1k",
                                dropout=args.dropout))
        outstream = "fc_drop"
        # Apply scaling
        proto.scaling(args.depth_multiplier)
        proto.set_bias_flag(False)

        proto.finalized('imagenet', outstream_name=outstream)
        proto.set_global_regularization(args.regularizer)
        proto.set_pretrain_path("/home/public/supernet-elasticnet")
        file = os.path.join(proto_dir, "elastic-subnet-depth-%d.prototxt" % id)

        with open(file, 'w') as fp:
            proto.dump(fp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, help="Budget to sample",
                        default=100)
    parser.add_argument("--yaml_cfg", type=str, nargs='?', help="YAML config for Backend settings.",
                        default=None)
    parser.add_argument("--depth_multiplier", type=float, help="Depth multiplier",
                        default=1.0)
    parser.add_argument("--regularizer", type=float, help="Regularizer",
                        default=1e-5)
    parser.add_argument("--dropout", type=float, help="Dropout rate",
                        default=0.0)
    parser.add_argument("--stem_activation_fn", type=str, help="Activation Function for head",
                        default="swish")
    parser.add_argument("--num_stem_convs", type=int, help="Number of filters in the head conv",
                        default=32)
    parser.add_argument("--head_activation_fn", type=str, help="Activation Function for head",
                        default="swish")
    parser.add_argument("--num_head_convs", type=int, help="Number of filters in the head conv",
                        default=1280)

    args = parser.parse_args()
    main(args)