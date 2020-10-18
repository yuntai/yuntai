import pandas as pd
import numpy as np
import scipy
from pathlib import Path
import scipy.integrate

def run(df, col, mid_freq=2409400000.0/1e7):
    vars_ = {}

    mid_ix = df.set_index('GHz').index.get_loc(mid_freq)

    x_coords = df['GHz'].values[[0, -1]]
    y_coords = df[col].values[[0, -1]]
    m, c = np.polyfit(x_coords, y_coords, 1)

    df['base_line'] = m * df['GHz'] + c
    df['val'] = df['base_line'] - df[col]

    P1Half = df.loc[:mid_ix, 'val']
    P2Half = df.loc[mid_ix:, 'val']
    P1_ix = P1Half.idxmax()
    P2_ix = P2Half.idxmax()

    P1 = df.loc[P1_ix, 'GHz']
    P2 = df.loc[P2_ix, 'GHz']

    Mm_L = df.loc[P1_ix, 'val']
    Mm_R = df.loc[P2_ix, 'val']

    vars_['Mm_L'] = Mm_L
    vars_['Mm_R'] = Mm_R

    def get_area(df, ix_l, ix_r, y='val', x='GHz'):
        a = df[y].loc[ix_l:ix_r]
        b = a.shift(-1)
        vals = (a.iloc[:-1] + b.iloc[:-1])/2
        return (vals * df.loc[ix_l:ix_r, 'GHz'].diff().iloc[1:]).sum()

    def get_lev_index(PHalf, Mm, lev, p_ix):
        LH, RH = PHalf.loc[:p_ix], PHalf.loc[p_ix:]
        ix_l = LH.searchsorted(Mm*lev, side='right')
        ix_r = RH.searchsorted(Mm*lev, side='left')
        return ix_l, ix_r

    for lev in [0.9, 0.6, 0.3]:
        ix_ll, ix_lr = get_lev_index(P1Half, Mm_L, lev, P1_ix)
        ix_rl, ix_rr = get_lev_index(P2Half, Mm_R, lev, P2_ix)
        a, b, c, d = df.loc[[ix_ll, ix_lr, ix_rl, ix_rr], 'GHz']
        vars_[f'{lev}_Mm_LL'] = a
        vars_[f'{lev}_Mm_LR'] = b
        vars_[f'{lev}_Mm_RL'] = c
        vars_[f'{lev}_Mm_RR'] = d
        if lev == 0.6:
            area_6l = get_area(df, ix_ll, ix_lr)
            area_6r = get_area(df, ix_rl, ix_rr)
    vars_['area_0.6l'] = area_6l
    vars_['area_0.6r'] = area_6r

    vars_['total_area'] = get_area(df, df.index[0], df.index[-1])
    vars_['p1_area'] = get_area(df, df.index[0], mid_ix)
    vars_['p2_area'] = get_area(df, mid_ix, df.index[-1])

    c1, *_ = np.polyfit(df.loc[[0,P1_ix], 'GHz'].values, df.loc[[0, P1_ix], 'val'].values, 1)
    c2, *_ = np.polyfit(df.loc[[P2_ix, df.index[-1]], 'GHz'].values, df.loc[[P2_ix, df.index[-1]], 'val'].values, 1)
    vars_['first_deriv_coeff_L'] = abs(c1)
    vars_['first_deriv_coeff_R'] = abs(c2)

    a, b, c, d = df['GHz'].values[[0, P1_ix, P2_ix, -1]]
    t1 = (b - a) * Mm_L / 2.
    t2 = (d - c) * Mm_R / 2.
    vars_['tri_area_L'] = t1
    vars_['tri_area_R'] = t2

    vars_['tri_area_prop'] = t1/t2
    vars_['coeff_prop'] = abs(c1)/abs(c2)

    # parabolic
    P1_ix_p = P1_ix * 2
    x_coords = df.loc[[0, P1_ix, P1_ix_p], 'GHz'].values
    y_coords = np.array([0, Mm_L, 0])
    eq = np.poly1d(np.polyfit(x_coords, y_coords, 2))
    parabolic_a_l, *_ = scipy.integrate.quad(eq, x_coords[0], x_coords[-1])
    vars_["2nd_coeff_L"] = eq[2]

    P2_ix_p = df.index[-1] - (df.index[-1] - P2_ix) * 2
    x_coords = df.loc[[P2_ix_p,P2_ix,df.index[-1]], 'GHz'].values
    y_coords = np.array([0, Mm_R, 0])
    eq = np.poly1d(np.polyfit(x_coords, y_coords, 2))
    parabolic_a_r, *_ = scipy.integrate.quad(eq, x_coords[0], x_coords[-1])
    vars_["2nd_coeff_R"] = eq[2]

    vars_["para_area_L"] = parabolic_a_l
    vars_["para_area_R"] = parabolic_a_r

    L1 = L2 = df.loc[P1_ix, 'GHz'] - df.loc[0, 'GHz']
    L3 = L4 = df.loc[df.shape[0]-1, 'GHz'] - df.loc[P2_ix, 'GHz']
    vars_['para_area_prop'] = parabolic_a_l/parabolic_a_r
    vars_['L1L3_prop'] = L1/L3
    return vars_

d=1e7
rootdir = Path('/mnt/tmp/rf')
fns = ['adenine.csv', 'cytosine.csv', 'thymine.csv', 'guanine.csv']
outfns = [fn.split('.')[0]+'_out.csv' for fn in fns]
for fn, outfn in zip(fns, outfns):
    df = pd.read_csv(rootdir/fn)
    df['GHz'] = df['GHz'].str.replace(',','').astype('float')/d
    cols = [c for c in df.columns if c != 'GHz']
    outdf = None
    for col in cols:
        vars_ = run(df, col)
        if outdf is None:
            outdf = pd.DataFrame()
            outdf["var"] = list(vars_.keys())
        outdf[col] = list(vars_.values())
    print(outdf)
    outdf.to_csv(rootdir/outfn, index=False)
