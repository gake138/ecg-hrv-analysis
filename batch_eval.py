"""批量评估：MIT-BIH全部48条记录 × 完整管线 → 每条Se/+P汇总表"""
import os
import csv
import numpy as np
import wfdb

from src.preprocess import preprocess_ecg
from src.detect import pan_tompkins_detect, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'mitdb')
RESULTS = os.path.join(ROOT, 'results')

# MIT-BIH心律失常数据库全部48条记录
RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
           111, 112, 113, 114, 115, 116, 117, 118, 119,
           121, 122, 123, 124,
           200, 201, 202, 203, 205, 207, 208, 209, 210,
           212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228,
           230, 231, 232, 233, 234]

# AAMI心拍类标注，排除 | ~ + [ ] 等非心拍标记
BEAT_SYMBOLS = list('NLRAaJSVFfejE/fQ')


def eval_record(rec):
    """单条记录全流程评估，返回指标dict"""
    path = f'{DATA_PATH}/{rec}'
    record = wfdb.rdrecord(path)
    fs = record.fs
    # 优先取MLII导联（参数按它调的），没有就退回第0通道
    ch = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
    ecg = record.p_signal[:, ch].astype(float)

    ann = wfdb.rdann(path, 'atr')
    syms = np.asarray(ann.symbol)
    r_ref = ann.sample[np.isin(syms, BEAT_SYMBOLS)]   # 所有心拍=真值

    ecg_filt = preprocess_ecg(ecg, fs)
    r_peaks = pan_tompkins_detect(ecg_filt, fs)
    tp, fp, fn, errs = evaluate(r_peaks, r_ref, fs)

    return {'rec': rec, 'tp': tp, 'fp': fp, 'fn': fn,
            'se': tp/(tp+fn)*100, 'pp': tp/(tp+fp)*100,
            'err': errs.mean() if len(errs) else float('nan')}


def run():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for rec in RECORDS:
        try:
            r = eval_record(rec)
            rows.append(r)
            print(f"record {r['rec']:>3}:  Se={r['se']:6.2f}%  +P={r['pp']:6.2f}%  "
                  f"TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
        except Exception as e:
            print(f"record {rec:>3}:  失败 ({e})")     # 单条失败不拖垮全批

    # 汇总：宏平均=每条记录一票；微平均=所有心拍一票
    se = np.array([r['se'] for r in rows]); pp = np.array([r['pp'] for r in rows])
    TP = sum(r['tp'] for r in rows); FP = sum(r['fp'] for r in rows); FN = sum(r['fn'] for r in rows)
    print('=' * 52)
    print(f'成功评估 {len(rows)}/{len(RECORDS)} 条')
    print(f'宏平均: Se={se.mean():.2f}%±{se.std():.1f}   +P={pp.mean():.2f}%±{pp.std():.1f}')
    print(f'微平均: Se={TP/(TP+FN)*100:.2f}%   +P={TP/(TP+FP)*100:.2f}%   (TP={TP} FP={FP} FN={FN})')

    # 存CSV（按Se升序=垫底的排最前），写报告直接用
    with open(f'{RESULTS}/batch_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['rec', 'se', 'pp', 'tp', 'fp', 'fn', 'err'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['se']):
            w.writerow(r)
    print('明细已存 results/batch_summary.csv')


if __name__ == '__main__':
    run()
