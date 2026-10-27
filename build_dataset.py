"""构建CNN数据集：从42条记录提取R波附近200点片段，按患者间划分存为npz"""
import os
import numpy as np
import wfdb

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'mitdb')
OUT  = os.path.join(ROOT, 'data')

HALF_BEFORE, HALF_AFTER = 99, 100

TRAIN_RECORDS = [100, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116,
                 121, 122, 123, 124,
                 201, 203, 205, 207, 208, 209, 213, 214, 215, 220, 221, 223, 228,
                 231, 232, 233, 234]
TEST_RECORDS  = [101, 117, 200, 202, 210, 212, 219, 222, 230]

AAMI = {'N':0,'L':0,'R':0,'e':0,'j':0,
        'A':1,'a':1,'J':1,'S':1,
        'V':2,'E':2,
        'F':3}
CLASS_NAMES = ['N', 'SVE', 'VE', 'F']


def segment_beats(rec):
    path = f'{DATA}/{rec}'
    record = wfdb.rdrecord(path)
    ch = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
    ecg = record.p_signal[:, ch].astype(float)
    ann = wfdb.rdann(path, 'atr')
    X, y = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI:
            continue
        if s - HALF_BEFORE < 0 or s + HALF_AFTER > len(ecg):
            continue
        seg = ecg[s - HALF_BEFORE: s + HALF_AFTER]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        X.append(seg); y.append(AAMI[sym])
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def save(records, name):
    X_all, y_all = [], []
    for r in records:
        X, y = segment_beats(r)
        X_all.append(X); y_all.append(y)
        print(f'record {r}: {len(y)} beats')
    X = np.concatenate(X_all); y = np.concatenate(y_all)
    np.savez_compressed(os.path.join(OUT, name), X=X, y=y)
    print(f'>>> {name}: 形状 {X.shape}, 各类数量 {dict(zip(CLASS_NAMES, np.bincount(y)))}')


if __name__ == '__main__':
    save(TRAIN_RECORDS, 'beats_train.npz')
    save(TEST_RECORDS,  'beats_test.npz')
