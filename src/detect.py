"""T3 R波检测模块：Pan-Tompkins五步算法 + 评估打分"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks


def pan_tompkins_detect(ecg_filt, fs):
    """Pan-Tompkins五步检测R波，返回R波顶点索引数组"""
    # ① 带通5-15Hz：只留QRS能量，压掉P/T波
    b, a = signal.butter(2, [5, 15], btype='bandpass', fs=fs)
    x = signal.filtfilt(b, a, ecg_filt)
    # ② 五点微分：突出QRS的陡峭斜率
    h = np.array([-1, -2, 0, 2, 1]) * fs / 8
    y = np.convolve(x, h, mode='same')
    # ③ 平方：全变正 + 拉大与噪声的对比度
    y = y ** 2
    # ④ 150ms滑窗积分：每个QRS聚成一个孤立鼓包
    w = int(0.15 * fs)
    y = np.convolve(y, np.ones(w)/w, mode='same')
    # ⑤ 阈值检测：数据驱动阈值 + 200ms不应期(生理极限心率300bpm)
    thr = 0.3 * np.percentile(y, 99)
    bumps, _ = find_peaks(y, height=thr, distance=int(0.2*fs))
    # 回映射：回到干净波形±50ms内找真实R波顶点(积分鼓包中心≠R顶点)
    half = int(0.05 * fs)
    r_peaks = np.array([p - half + np.argmax(ecg_filt[max(p-half, 0):p+half+1])
                        for p in bumps])
    return r_peaks


def evaluate(det, ref, fs, tol_ms=150):
    """对标专家标注打分，返回 (tp, fp, fn, 误差ms数组)"""
    tol = int(tol_ms/1000 * fs)
    matched = np.zeros(len(ref), bool)
    tp = fp = 0; errs = []
    for p in det:
        j = np.argmin(np.abs(ref - p))       # 找最近的真值心跳
        if abs(ref[j] - p) <= tol and not matched[j]:
            tp += 1; matched[j] = True       # 命中，且该真值只能被用一次
            errs.append((p - ref[j])/fs*1000)
        else:
            fp += 1                          # 匹配不上 = 误检
    fn = int((~matched).sum())               # 没被匹配到的真值 = 漏检
    return tp, fp, fn, np.array(errs)


if __name__ == '__main__':
    # 自测1：合成尖峰信号，验证检测管线
    fs = 360
    t = np.arange(0, 10, 1/fs)
    fake = np.zeros(len(t))
    fake[int(0.5*fs)::int(1*fs)] = 1.0
                     # 每秒一个尖峰
    peaks = pan_tompkins_detect(fake, fs)
    print(f'自测1：10秒假信号检出{len(peaks)}个峰（期望10）')

    # 自测2：拿检测结果当"真值"喂给evaluate，完美系统必然全对
    tp, fp, fn, errs = evaluate(peaks, peaks.copy(), fs)
    print(f'自测2：TP={tp} FP={fp} FN={fn}（期望10/0/0），误差={errs.mean():.1f}ms（期望0.0）')
