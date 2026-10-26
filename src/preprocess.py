"""T2 滤波去噪模块：0.5Hz高通 + 50Hz陷波 + 40Hz低通（零相位）"""
from scipy import signal


def design_filters(fs):
    """设计三段滤波器系数（只算系数，不碰数据）"""
    b_hp, a_hp       = signal.butter(2, 0.5, btype='highpass', fs=fs)  # 去基线漂移
    b_notch, a_notch = signal.iirnotch(50, Q=30, fs=fs)                # 去50Hz工频
    b_lp, a_lp       = signal.butter(2, 40, btype='lowpass', fs=fs)    # 去肌电高频
    return (b_hp, a_hp), (b_notch, a_notch), (b_lp, a_lp)


def preprocess_ecg(ecg, fs):
    """三段零相位滤波链，返回与输入等长的滤波后信号"""
    (b_hp, a_hp), (b_notch, a_notch), (b_lp, a_lp) = design_filters(fs)
    x = signal.filtfilt(b_hp,    a_hp,    ecg)   # filtfilt=零相位，不移动波形位置
    x = signal.filtfilt(b_notch, a_notch, x)
    x = signal.filtfilt(b_lp,    a_lp,    x)
    return x


if __name__ == '__main__':
    # 直接运行本文件时才执行的自测（被import时跳过）
    import numpy as np
    fs = 360
    t = np.arange(0, 10, 1/fs)
    fake = (np.sin(2*np.pi*1.2*t) + 2*np.sin(2*np.pi*0.3*t)
            + 0.5*np.sin(2*np.pi*50*t))   # 心跳+漂移+工频的合成假信号
    out = preprocess_ecg(fake, fs)
    print('自测通过：输入长度', len(fake), '输出长度', len(out))
