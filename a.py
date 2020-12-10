import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

dataroot = Path('/mnt/datasets/NIPA/comp_2020')
train_df = pd.read_csv(dataroot/'train/train.tsv', sep='\t', header=None)
train_df.columns = ['fn', 'l0', 'l1']
train_df['label'] = train_df.groupby(['l0','l1']).ngroup()

SEED=2003
labels = train_df.label.values
pl.trainer.seed_everything(SEED)

skf = StratifiedKFold(shuffle=True)
inds = []
for fold_i, (train_ind, val_ind) in enumerate(skf.split(np.zeros_like(labels), labels)):
    print(np.array(val_ind).mean())

print("..")
skf = StratifiedKFold(shuffle=True)
inds = []
for fold_i, (train_ind, val_ind) in enumerate(skf.split(np.zeros_like(labels), labels)):
    print(np.array(val_ind).mean())
