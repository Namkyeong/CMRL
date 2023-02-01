import numpy as np

def write_summary(args, config_str, stats):
    
    f = open("results/bias_{}/{}.txt".format(args.bias, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("ROC : {:.4f} || AP : {:.4f} || F1 : {:.4f} || Acc : {:.4f} ".format(stats[0], stats[1], stats[2], stats[3]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    f = open("results/bias_{}/{}_total.txt".format(args.bias, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("ROC : {:.4f}({:.4f}) || AP : {:.4f}({:.4f}) || F1 : {:.4f}({:.4f}) || Acc : {:.4f}({:.4f}) ".format(stats[0], stats[1], stats[2], stats[3],
                                                                                                                                    stats[4], stats[5], stats[6], stats[7]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name in ['embedder', 'lr', 'lam1', 'lam2', 'bias'
                    ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]