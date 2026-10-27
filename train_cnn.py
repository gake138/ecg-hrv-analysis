"""1D-CNN心拍分类：4类(N/SVE/VE/F)，患者间划分，含类别加权与混淆矩阵"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, BATCH, LR = 50, 128, 1e-3
CLASS_NAMES = ['N', 'SVE', 'VE', 'F']


class CNN1D(nn.Module):
    def __init__(self, n_class=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_class)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))


def load(split):
    d = np.load(os.path.join(ROOT, 'data', f'beats_{split}.npz'))
    X = torch.from_numpy(d['X']).unsqueeze(1)
    y = torch.from_numpy(d['y'])
    assert X.shape[0] > 10000, f'【数据集异常】{split}只有{X.shape[0]}拍，build_dataset.py没跑对，禁止训练！'
    return X, y


def evaluate(model, loader):
    model.eval()
    cm = np.zeros((4, 4), dtype=int)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                cm[t, p] += 1
    print('混淆矩阵 (行=真实, 列=预测)  类别顺序:', CLASS_NAMES)
    print(cm)
    print('\n每类指标:')
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        se = tp / (tp + fn) if tp + fn else 0
        pp = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * se * pp / (se + pp) if se + pp else 0
        f1s.append(f1)
        print(f'  {name:>4}: Se={se*100:6.2f}%   +P={pp*100:6.2f}%   F1={f1*100:6.2f}%')
    acc = cm.trace() / cm.sum()
    print(f'\n总体 accuracy = {acc*100:.2f}%   macro-F1 = {np.mean(f1s)*100:.2f}%')


def main():
    Xtr, ytr = load('train'); Xte, yte = load('test')
    print(f'训练 {len(ytr)} 拍, 测试 {len(yte)} 拍, device = {DEVICE}')

    counts = np.bincount(ytr.numpy(), minlength=4)
    weights = torch.tensor((counts.sum() / (4.0 * counts)) ** 0.5,
                           dtype=torch.float32).to(DEVICE)
    print('类权重:', dict(zip(CLASS_NAMES, weights.cpu().numpy().round(2))))

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=BATCH)

    model = CNN1D().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=weights)

    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        print(f'epoch {ep:>2}/{EPOCHS}   loss = {tot/len(ytr):.4f}')
        sched.step()

    torch.save(model.state_dict(), os.path.join(ROOT, 'cnn_weights.pt'))
    print('\n=== 测试集结果 ===')
    evaluate(model, te_loader)


if __name__ == '__main__':
    main()
