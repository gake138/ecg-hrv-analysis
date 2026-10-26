"""全流程入口：读取 → 滤波 → 检测 → HRV → 输出指标"""
"""全流程入口：读取 → 滤波 → 检测 → 评估 → HRV → 打印指标 + 生成四张图"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')        # 只存文件不弹窗，保证任何环境可复现
import matplotlib.pyplot as plt
import wfdb

from src.preprocess import preprocess_ecg
from src.detect import pan_tompkins_detect, evaluate
from src.hrv import compute_hrv

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'mitdb')
RESULTS = os.path.join(ROOT, 'results')
REC = '100'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def make_figures(ecg, ecg_filt, fs, r_peaks, r_ref, res, rec):
    os.makedirs(RESULTS, exist_ok=True)

    # 图1 滤波对比（前10秒）
    n = 10 * fs
    t = np.arange(n) / fs
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, ecg[:n], color='gray')
    ax[0].set_title(f'Raw ECG (rec {rec})')
    ax[1].plot(t, ecg_filt[:n], color='tab:blue')
    ax[1].set(title='Filtered (0.5Hz HP + 50Hz notch + 40Hz LP)', xlabel='Time (s)')
    fig.tight_layout(); fig.savefig(f'{RESULTS}/filter_compare.png', dpi=150); plt.close(fig)

    # 图2 检测 vs 专家标注（前30秒）
    n = 30 * fs
    t = np.arange(n) / fs
    det_in = r_peaks[r_peaks < n]
    ref_in = r_ref[r_ref < n]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ecg_filt[:n], color='tab:blue', lw=0.8, label='ECG filtered')
    ax.plot(det_in / fs, ecg_filt[det_in], 'rv', ms=8, label=f'detected ({len(det_in)})')
    ax.plot(ref_in / fs, ecg_filt[ref_in], 'x', color='orange', ms=8, mew=2,
            label=f'annotation ({len(ref_in)})')
    ax.set(title='R-peak detection vs expert annotation (first 30 s)', xlabel='Time (s)')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{RESULTS}/detection_check.png', dpi=150); plt.close(fig)

    # 图3 Tachogram + Poincaré
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(res['t'], res['rr'], lw=0.5)
    ax[0].set(xlabel='Time (s)', ylabel='RR (ms)', title='心动周期序列 Tachogram')
    ax[1].scatter(res['rr'][:-1], res['rr'][1:], s=4, alpha=0.4)
    ax[1].set(xlabel='RR$_n$ (ms)', ylabel='RR$_{n+1}$ (ms)', title='Poincaré图：沿对角线的椭圆')
    fig.tight_layout(); fig.savefig(f'{RESULTS}/tachogram_poincare.png', dpi=150); plt.close(fig)

    # 图4 HRV频谱（对数坐标 + LF/HF着色）
    f, p = res['f'], res['psd']
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(f, p, lw=1.2)
    ax.axvspan(0.04, 0.15, color='orange', alpha=0.18, label=f'LF={res["lf"]:.0f}')
    ax.axvspan(0.15, 0.40, color='green', alpha=0.18, label=f'HF={res["hf"]:.0f}')
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (ms²/Hz)', title='HRV频谱', xlim=(0, 0.5))
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{RESULTS}/hrv_spectrum.png', dpi=150); plt.close(fig)


def run(rec=REC):
    # ---- T1 读取（真值只取N拍，与你notebook一致）----
    record = wfdb.rdrecord(f'{DATA_PATH}/{rec}')
    ann = wfdb.rdann(f'{DATA_PATH}/{rec}', 'atr')
    fs = record.fs
    ecg = record.p_signal[:, 0].astype(float)
    syms = np.asarray(ann.symbol)
    beat = np.isin(syms, list('NLRAaJSVFfejE/fQ'))   # 所有心拍类标注(AAMI)，排除+|~等非心拍
    print(f'标注检查: 共{len(syms)}条, N拍{int((syms=="N").sum())}个, 心拍共{int(beat.sum())}个')
    r_ref = ann.sample[beat]


    # ---- T2 滤波 ----
    ecg_filt = preprocess_ecg(ecg, fs)

    # ---- T3 检测 + 评估 ----
    r_peaks = pan_tompkins_detect(ecg_filt, fs)
    tp, fp, fn, errs = evaluate(r_peaks, r_ref, fs)

    # ---- T4 HRV ----
    res = compute_hrv(r_peaks, fs)

    # ---- 打印汇总 ----
    print('=' * 48)
    print(f' ECG项目全流程结果  record {rec}')
    print('=' * 48)
    print(f'【T2】滤波完成  长度={len(ecg_filt)}  采样率={fs}Hz')
    print(f'【T3】TP={tp}  FP={fp}  FN={fn}')
    print(f'     Se ={tp/(tp+fn)*100:.2f}%   +P={tp/(tp+fp)*100:.2f}%')
    print(f'     定位误差={errs.mean():+.1f}±{errs.std():.1f} ms')
    print(f'【T4】平均RR={res["mean_rr"]:.0f}ms ({60000/res["mean_rr"]:.0f} bpm)')
    print(f'     SDNN={res["sdnn"]:.1f}ms  RMSSD={res["rmssd"]:.1f}ms  pNN50={res["pnn50"]:.1f}%')
    print(f'     Poincaré: SD1={res["sd1"]:.1f}  SD2={res["sd2"]:.1f}  SD1/SD2={res["sd1_sd2"]:.2f}')
    print(f'     LF={res["lf"]:.0f}ms²  HF={res["hf"]:.0f}ms²  LF/HF={res["lf_hf"]:.2f}')

    make_figures(ecg, ecg_filt, fs, r_peaks, r_ref, res, rec)
    print(f'\n四张图已存入 results/')
    return res


if __name__ == '__main__':
    run()
