""" 
2022.5.5
From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/summary.py
Summary utilities
Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass

def get_outdir(path, *paths, inc=False):
    r"""
    创建目录，如果目录存在，则自动加 1 
    参数:
        path (:obj:`str`):
            目录路径.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.

    """
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    r"""
    每个Epoch后打印日志
    参数:
        epoch (:obj:`int`):
            Epoch.
        train_metrics (:obj:`dict`):
            训练结果.
        eval_metrics (:obj:`dict`):
            测试结果.
        filename (:obj:`str`):
            日志文件名.
        write_header (:obj:`bool`, `optional`, defaults to :obj:`False`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        log_wandb (:obj:`bool`, `optional`, defaults to :obj:`False`):
            是否使用wandb.
    """
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)