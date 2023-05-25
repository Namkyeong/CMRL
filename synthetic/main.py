# from train import train_baseline_syn
# from train_causal import train_causal_syn
from opts import setup_seed
import torch
import opts
import os
import utils
import pdb
import time
import warnings
import report
warnings.filterwarnings('ignore')

def experiment():

    args = opts.parse_args()
    save_path = "data"
    os.makedirs(save_path, exist_ok=True)
    try:
        dataset = torch.load(save_path + "/syn_dataset.pt")
    except:
        dataset = utils.graph_dataset_generate(args, save_path)
    
    try:
        paired_dataset = torch.load(save_path + "/syn_pair_dataset.pt")
    except:
        paired_dataset = utils.pair_dataset_generate(dataset, save_path)
    
    train_set, valid_set, test_set = utils.pair_dataset_bias_split(paired_dataset, args, bias=args.bias, split=[6, 2, 2])

    best_rocs, best_aps, best_f1s, best_accs = [], [], [], []
    
    for repeat in range(1, args.repeat + 1):
    
        stats, config_str, _, _ = main(args, train_set, valid_set, test_set, repeat = repeat)
        
        # get Stats
        best_rocs.append(stats[0])
        best_aps.append(stats[1])
        best_f1s.append(stats[2])
        best_accs.append(stats[3])

        report.write_summary(args, config_str, stats)
    
    roc_mean, roc_std = report.get_stats(best_rocs)
    ap_mean, ap_std = report.get_stats(best_aps)
    f1_mean, f1_std = report.get_stats(best_f1s)
    accs_mean, accs_std = report.get_stats(best_accs)

    report.write_summary_total(args, config_str, [roc_mean, roc_std, ap_mean, ap_std, f1_mean, f1_std, accs_mean, accs_std])


def main(args, train_df, valid_df, test_df, repeat = 0, fold = 0):

    if args.embedder == 'CMRL':
        from models import CMRL_ModelTrainer
        embedder = CMRL_ModelTrainer(args, train_df, valid_df, test_df, repeat, fold)

    best_roc, best_ap, best_f1, best_acc = embedder.train()

    return [best_roc, best_ap, best_f1, best_acc], embedder.config_str, embedder.best_config_roc, embedder.best_config_f1


if __name__ == '__main__':
    experiment()
