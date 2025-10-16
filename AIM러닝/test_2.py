# ============================================
# PyTorch MLP로 회귀 문제 풀기 (study_time, rest_time, condition → score)
# - CSV: study_data.csv (UTF-8-SIG)
# - 열: study_time, rest_time, condition, score
# - 기능: 표준화, Train/Val/Test 분할, MLP(2은닉층), EarlyStopping, 저장/로드
# - 지표: MAE, RMSE, R2
# ============================================
import os
import json
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------------------------
# 0) 기본 설정 (재현성 + 디바이스)
# -------------------------------------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 재현성(완전 고정은 성능 다소 하락 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------------------------
# 1) 데이터 로드
# -------------------------------------------------
CSV_PATH = "study_data.csv"
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 컬럼명 안전 처리
df.columns = [c.strip().lower() for c in df.columns]
required = {"study_time", "rest_time", "condition", "score"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV에 필요한 열 {required} 이(가) 없습니다. 실제 열: {df.columns.tolist()}")

# 특징(X), 타깃(y)
X = df[["study_time", "rest_time", "condition"]].to_numpy(np.float32)
y = df[["score"]].to_numpy(np.float32)  # (N, 1) 모양을 유지

# -------------------------------------------------
# 2) Train/Val/Test 분할 (8:1:1)
# -------------------------------------------------
def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, seed=SEED):
    N = len(X)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)

# -------------------------------------------------
# 3) 표준화(Standardization)
#    - 반드시 'Train 통계'로만 fit → Val/Test에 동일 적용
# -------------------------------------------------
class StandardScaler:
    def fit(self, arr: np.ndarray):
        self.mean_ = arr.mean(axis=0, keepdims=True)
        self.std_  = arr.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, arr: np.ndarray):
        return (arr - self.mean_) / self.std_
    def inverse_transform(self, arr: np.ndarray):
        return arr * self.std_ + self.mean_

# 입력 X, 타깃 y 모두 스케일링(회귀에서 학습 안정성↑)
x_scaler = StandardScaler().fit(X_tr)
y_scaler = StandardScaler().fit(y_tr)

X_tr_s  = x_scaler.transform(X_tr)
X_val_s = x_scaler.transform(X_val)
X_te_s  = x_scaler.transform(X_te)

y_tr_s  = y_scaler.transform(y_tr)
y_val_s = y_scaler.transform(y_val)
y_te_s  = y_scaler.transform(y_te)

# -------------------------------------------------
# 4) Tensor 변환 + DataLoader 구성
# -------------------------------------------------
BATCH_SIZE = 512  # 10만행이면 256~4096 사이에서 실험하며 조정

tr_ds  = TensorDataset(torch.from_numpy(X_tr_s),  torch.from_numpy(y_tr_s))
val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s))
te_ds  = TensorDataset(torch.from_numpy(X_te_s),  torch.from_numpy(y_te_s))

# num_workers는 OS/환경에 맞게 조절(Windows면 0 권장, Linux면 CPU 코어수-1 정도)
tr_loader  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
te_loader  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

# -------------------------------------------------
# 5) 모델 정의(MLP)
#    - 은닉 2층 (64, 64), ReLU
#    - 출력 1 (회귀이므로 활성함수 없음)
# -------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int]=(64, 64), dropout=0.0):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)  # 출력 1
        )

        # He/Xavier 초기화(선택) – PyTorch 기본초기화도 충분히 좋음
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=3, hidden=(64, 64), dropout=0.05).to(DEVICE)

# -------------------------------------------------
# 6) 손실함수, 옵티마이저, 스케줄러(선택)
# -------------------------------------------------
criterion = nn.MSELoss()  # 회귀 → MSE
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)  # L2는 weight_decay
# 학습률 스케줄러(선택): Plateau에서 LR 줄이기
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=5)

# -------------------------------------------------
# 7) 학습 루프 + Early Stopping
# -------------------------------------------------
EPOCHS   = 200
PATIENCE = 20  # 검증 손실이 감소하지 않으면 조기 종료

best_val = float('inf')
best_state = None
wait = 0

def evaluate_mse(dl):
    """현재 모델로 데이터로더 mse 반환(스케일된 y 기준)"""
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            bs = yb.size(0)
            mse_sum += loss.item() * bs
            n += bs
    return mse_sum / max(n, 1)

# --- 학습 곡선 기록용 ---
history_train = []
history_val   = []


for epoch in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()
    train_loss_sum, n_tr = 0.0, 0
    for xb, yb in tr_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)                # 순전파
        loss = criterion(pred, yb)      # 손실(MSE)
        loss.backward()                 # 역전파(autograd)
        optimizer.step()                # 가중치 갱신

        bs = yb.size(0)
        train_loss_sum += loss.item() * bs
        n_tr += bs

    train_mse = train_loss_sum / max(n_tr, 1)

    # ---- Validation ----
    val_mse = evaluate_mse(val_loader)
    scheduler.step(val_mse)  # Plateau면 LR 감소

    if epoch == 1 or epoch % 10 == 0:
        print(f"[{epoch:03d}] train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")

    # ---- Early Stopping ----
    if val_mse + 1e-8 < best_val:
        best_val = val_mse
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping @ epoch {epoch}, best val_mse={best_val:.6f}")
            break
    history_train.append(train_mse)
    history_val.append(val_mse)


# 베스트 파라미터로 복원
if best_state is not None:
    model.load_state_dict(best_state)

# -------------------------------------------------
# 8) 평가 지표(원 스케일로 복원 후 계산)
# -------------------------------------------------
def to_numpy(t):
    return t.detach().cpu().numpy()

def inverse_target(y_scaled: np.ndarray) -> np.ndarray:
    return y_scaled * y_scaler.std_ + y_scaler.mean_

@torch.no_grad()
def predict_loader(dl) -> Tuple[np.ndarray, np.ndarray]:
    """스케일 복원된 (y_true, y_pred) 반환"""
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        ys.append(to_numpy(yb))
        ps.append(to_numpy(pred))
    y_scaled = np.vstack(ys)
    p_scaled = np.vstack(ps)
    # 원 스케일로 복원
    y_true = inverse_target(y_scaled)
    y_pred = inverse_target(p_scaled)
    return y_true, y_pred

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

# Train/Val/Test 성능
for name, loader in [("Train", tr_loader), ("Val", val_loader), ("Test", te_loader)]:
    yt, yp = predict_loader(loader)
    print(f"[{name}] MAE={mae(yt, yp):.4f} | RMSE={rmse(yt, yp):.4f} | R2={r2_score(yt, yp):.4f}")



# -------------------------------------------------
# 9) 단일 샘플 추론 예시 (원시 입력 → 스케일 → 예측 → 복원)
# -------------------------------------------------
def predict_single(study_time: float, rest_time: float, condition: float) -> float:
    x = np.array([[study_time, rest_time, condition]], dtype=np.float32)
    x_s = (x - x_scaler.mean_) / x_scaler.std_
    with torch.no_grad():
        xt = torch.from_numpy(x_s).to(DEVICE)
        pred_scaled = model(xt).cpu().numpy()  # 스케일된 예측
    pred = inverse_target(pred_scaled)         # 원 스케일로 복원
    return float(pred.squeeze())

ex_pred = predict_single(4.0, 3.0, 6.0)
print(f"예시 예측(4h 공부, 휴식 3, 컨디션 6) → 예상 score ≈ {ex_pred:.2f}")

# -------------------------------------------------
# 10) 모델/스케일러 저장 & 로드
# -------------------------------------------------
SAVE_DIR = "artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

# 모델 가중치 저장
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mlp_regressor.pt"))

# ============================================
# 시각화 (matplotlib만 사용, 색 지정 X, 그림 1개씩)
# ============================================
import matplotlib.pyplot as plt
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- (1) 학습 곡선: Train vs Val MSE ----------
plt.figure()
plt.plot(range(1, len(history_train)+1), history_train, label="Train MSE")
plt.plot(range(1, len(history_val)+1),   history_val,   label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Learning Curve (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "01_learning_curve_mse.png"))
plt.close()

# ---------- (2) Test: y_true vs y_pred (Parity Plot) ----------
# 테스트 예측(이미 위에서 계산했으니 재사용 가능하지만, 안전하게 다시 받자)
yt_test, yp_test = predict_loader(te_loader)

plt.figure()
plt.scatter(yt_test, yp_test, s=10, alpha=0.7)
# y = x 기준선
min_v = float(min(yt_test.min(), yp_test.min()))
max_v = float(max(yt_test.max(), yp_test.max()))
plt.plot([min_v, max_v], [min_v, max_v])
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Test Parity Plot (True vs Pred)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "02_test_parity_true_vs_pred.png"))
plt.close()

# ---------- (3) Test: Residual Histogram ----------
residuals = (yp_test - yt_test).reshape(-1)

plt.figure()
plt.hist(residuals, bins=50)
plt.xlabel("Residual (Pred - True)")
plt.ylabel("Count")
plt.title("Test Residuals Histogram")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "03_test_residuals_hist.png"))
plt.close()

# ---------- (4) Test: Residuals vs Predicted ----------
plt.figure()
plt.scatter(yp_test, residuals, s=10, alpha=0.7)
plt.axhline(0.0)
plt.xlabel("Predicted Score")
plt.ylabel("Residual (Pred - True)")
plt.title("Residuals vs Predicted (Test)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "04_residuals_vs_predicted_test.png"))
plt.close()

print("시각화 이미지 저장 완료 →", SAVE_DIR)

# 스케일러 통계 저장(나중에 추론시 동일 전처리 필요)
scaler_blob = {
    "x_mean": x_scaler.mean_.tolist(),
    "x_std":  x_scaler.std_.tolist(),
    "y_mean": y_scaler.mean_.tolist(),
    "y_std":  y_scaler.std_.tolist(),
}
with open(os.path.join(SAVE_DIR, "scalers.json"), "w", encoding="utf-8") as f:
    json.dump(scaler_blob, f, ensure_ascii=False, indent=2)

print("모델과 스케일러 통계를 저장했습니다:", SAVE_DIR)

# (참고) 로드시:
# model = MLPRegressor(in_dim=3, hidden=(64, 64), dropout=0.05).to(DEVICE)
# model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "mlp_regressor.pt"), map_location=DEVICE))
# with open(os.path.join(SAVE_DIR, "scalers.json"), "r", encoding="utf-8") as f:
#     blob = json.load(f)
#     x_scaler.mean_ = np.array(blob["x_mean"], dtype=np.float32)
#     x_scaler.std_  = np.array(blob["x_std"],  dtype=np.float32)
#     y_scaler.mean_ = np.array(blob["y_mean"], dtype=np.float32)
#     y_scaler.std_  = np.array(blob["y_std"],  dtype=np.float32)
