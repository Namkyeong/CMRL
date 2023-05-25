import argparse
import numpy as np
import random
import torch
from itertools import product

def parse_args():
    
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--embedder', type=str, default="CMRL")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--es", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval_freq", type=int, default=1)

    #################### toy example #######################
    parser.add_argument('--pretrain', type=int, default=30)
    parser.add_argument('--data_num', type=int, default=2000)
    parser.add_argument('--node_num', type=int, default=15)
    parser.add_argument('--max_degree', type=int, default=10)
    parser.add_argument('--feature_dim', type=int, default=-1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--shape_num', type=int, default=1)
    parser.add_argument('--bias', type=float, default=0.5)
    parser.add_argument('--penalty_weight', default=0.1, type=float, help='penalty weight')
    parser.add_argument('--train_type', type=str, default="base", help="irm, dro, base")

    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--the', type=int, default=0)
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--eval_random', type=str2bool, default=False)
    parser.add_argument('--normalize', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--inference', type=str2bool, default=False)
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    
    parser.add_argument('--k', type=int, default=3)

    #################### Causal GNN settings #######################
    if 'CAMPS' in parser.parse_known_args()[0].embedder:    
        parser.add_argument("--intervention", action = "store_true", default=False, help = 'Do Causal Intervention?')
        parser.add_argument("--conditional", action = "store_true", default=False, help = 'Do Conditional Causal Intervention?')
        parser.add_argument("--lam1", type=float, default=1.0)
        parser.add_argument("--lam2", type=float, default=1.0)

    ##################### baseline training ######################
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--fc_num', type=str, default="222")
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--save_dir', type=str, default="debug")
    parser.add_argument('--dataset', type=str, default="NCI1")
    parser.add_argument('--epoch_select', type=str, default='test_max')
    parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN")
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--global_pool', type=str, default="sum")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_checkpoints", action = "store_true", default=False, help = 'cross validation when True')
    parser.add_argument("--writer", action = "store_true", default=False, help = 'Tensorboard writer')

    args = parser.parse_args()
    print_args(args)
    setup_seed(args.seed)
    return args

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_n_filter_triples(datasets, 
                            feat_strs=['deg+odeg100'], 
                            nets=['ResGCN'], 
                            gfn_add_ak3=True,
                            gfn_reall=True, 
                            reddit_odeg10=True,
                            dd_odeg10_ak1=True):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered