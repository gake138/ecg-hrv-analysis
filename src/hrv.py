"""T4 HRV分析模块：时域指标 + Poincaré + 频域LF/HF"""
"""T4 HRV分析模块：RR清洗 + 时域指标 + Poincaré + 频域LF/HF"""
import numpy as np
from scipy import signal


def clean_rr(r_peaks, fs):
    """从R波位置生成RR间期并剔除伪迹
    规则：生理区间300-2000ms + 偏离中位数<20%（与notebook Cell 10一致）"""
    rr = np.diff(r_peaks) / fs * 1000.0      # ms
    t = r_peaks[1:] / fs                      # 每拍时刻（秒）
    ok = (rr > 300) & (rr < 2000)
    ok &= np.abs(rr - np.median(rr)) < 0.2 * np.median(rr)
    return rr[ok], t[ok]


def time_domain(rr):
    """时域指标：SDNN / RMSSD / pNN50 / 平均RR"""
    diffs = np.diff(rr)
    return {
        'mean_rr': rr.mean(),
        'sdnn': rr.std(),
        'rmssd': np.sqrt((diffs ** 2).mean()),
        'pnn50': (np.abs(diffs) > 50).mean() * 100,
    }


def poincare(rr):
    """Poincaré短长轴：SD1≈副交感，SD2≈整体慢变异"""
    d1 = (rr[1:] - rr[:-1]) / np.sqrt(2)      # 垂直对角线方向
    d2 = (rr[1:] + rr[:-1]) / np.sqrt(2)      # 沿对角线方向
    return {'sd1': d1.std(), 'sd2': d2.std(), 'sd1_sd2': d1.std() / d2.std()}


def spectrum(rr, t, fs_hr=4, nperseg=1024):
    """非均匀RR序列 → 4Hz均匀插值 → Welch频谱 → LF/HF功率"""
    t_grid = np.arange(t[0], t[-1], 1 / fs_hr)
    rr_i = np.interp(t_grid, t, rr)
    rr_i = rr_i - rr_i.mean()                 # 去均值再FFT
    f, p = signal.welch(rr_i, fs=fs_hr, nperseg=nperseg)
    lf = p[(f >= 0.04) & (f < 0.15)].sum()    # ← 若你Cell 13用的是np.trapz，换成你的写法
    hf = p[(f >= 0.15) & (f < 0.40)].sum()
    return {'f': f, 'psd': p, 'lf': lf, 'hf': hf, 'lf_hf': lf / hf}


def compute_hrv(r_peaks, fs):
    """一键算全套HRV指标，返回dict"""
    rr, t = clean_rr(r_peaks, fs)
    out = {'rr': rr, 't': t}
    out.update(time_domain(rr))
    out.update(poincare(rr))
    out.update(spectrum(rr, t))
    return out


if __name__ == '__main__':
    # 自测：合成"800ms心跳 + 0.25Hz呼吸正弦(30ms) + 3ms噪声"的30分钟假RR序列
    fs = 360
    rng = np.random.default_rng(42)
    beat_t = [0.0]
    while beat_t[-1] < 1800:
        rr_ms = 800 + 30 * np.sin(2 * np.pi * 0.25 * beat_t[-1]) + rng.normal(0, 3)
        beat_t.append(beat_t[-1] + rr_ms / 1000)
    r_peaks = (np.array(beat_t[:-1]) * fs).astype(int)

    res = compute_hrv(r_peaks, fs)
    f, p = res['f'], res['psd']
    band = (f >= 0.15) & (f < 0.40)
    f_hf = f[band][np.argmax(p[band])]
    print(f"自测: 平均RR={res['mean_rr']:.0f}ms (期望≈800)")
    print(f"     SDNN={res['sdnn']:.1f}ms (期望≈21)")
    print(f"     SD1={res['sd1']:.1f}ms (期望≈19)")
    print(f"     HF峰={f_hf:.3f}Hz (期望≈0.250 —— 呼吸频率被频谱正确还原)")
