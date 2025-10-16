import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ----------------------------
# 0) 데이터
# ----------------------------
df = pd.read_csv("UCI_Credit_Card.csv")
TARGET = "default.payment.next.month"
X_full = df.drop(columns=["ID", TARGET]).astype(np.float32)
y_full = df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
ALL_FEATURES = X_full.columns.tolist()

# 고정된 Stratified 9-Fold 분할 (전 라운드 재사용)
SK = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
SPLITS = list(SK.split(X_full.values, y_full.ravel()))

# 클래스 가중치 (양성에 더 큰 가중)
n_pos = float((y_full == 1).sum())
n_neg = float((y_full == 0).sum())
POS_WEIGHT = n_neg / max(1.0, n_pos)  # 불균형 보정

# ----------------------------
# 1) 로지스틱 + GD(모멘텀/감쇠) + 가중 BCE
# ----------------------------
def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))

def bce_loss_weighted(y_true, y_prob, pos_weight=1.0, eps=1e-8):
    y = y_true.ravel().astype(np.float32)
    p = np.clip(y_prob.ravel().astype(np.float32), eps, 1 - eps)
    w = np.where(y == 1.0, pos_weight, 1.0).astype(np.float32)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean() / w.mean())

def train_logistic_gd(
    X, y, epochs=1000, lr=0.05, l2=1e-4, pos_weight=1.0,
    momentum=0.9, lr_decay_patience=10, lr_decay=0.5
):
    m, d = X.shape
    w = np.zeros((d, 1), dtype=np.float32)
    b = 0.0

    vw = np.zeros_like(w)   # momentum buffer
    vb = 0.0

    w_sample = np.where(y.ravel() == 1.0, pos_weight, 1.0).astype(np.float32).reshape(-1, 1)
    norm = w_sample.mean()

    best_loss = np.inf
    bad_streak = 0
    cur_lr = lr

    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)

        # 가중 그라디언트
        grad = (w_sample * (p - y)) / norm
        dw = (X.T @ grad) / m + (l2 / m) * w
        db = grad.mean()

        # 모멘텀 업데이트
        vw = momentum * vw + (1 - momentum) * dw
        vb = momentum * vb + (1 - momentum) * db
        w -= cur_lr * vw
        b -= cur_lr * vb

        # 간단한 lr 감쇠(훈련 BCE 기준)
        train_loss = bce_loss_weighted(y, p, pos_weight=pos_weight)
        if train_loss + 1e-7 < best_loss:
            best_loss = train_loss
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= lr_decay_patience:
                cur_lr *= lr_decay
                bad_streak = 0

    return w, b

# ----------------------------
# 2) 9-Fold 학습/평가 (가중 BCE + AUC)
# ----------------------------
def kfold_train_models(feature_list, epochs=1000, lr=0.05, l2=1e-4, pos_weight=POS_WEIGHT):
    X = X_full[feature_list].to_numpy(dtype=np.float32)
    y = y_full.astype(np.float32)

    folds, bces, aucs = [], [], []

    for tr_idx, va_idx in SPLITS:  # 고정된 분할
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw).astype(np.float32)
        X_va = scaler.transform(X_va_raw).astype(np.float32)

        w, b = train_logistic_gd(
            X_tr, y_tr, epochs=epochs, lr=lr, l2=l2, pos_weight=pos_weight,
            momentum=0.9, lr_decay_patience=10, lr_decay=0.5
        )
        p_va = sigmoid(X_va @ w + b)

        L = bce_loss_weighted(y_va, p_va, pos_weight=pos_weight)
        try:
            auc = roc_auc_score(y_va.ravel(), p_va.ravel())
        except ValueError:
            auc = 0.5

        folds.append({"scaler": scaler, "w": w, "b": b, "X_va": X_va, "y_va": y_va, "loss_va": L, "auc_va": auc})
        bces.append(L); aucs.append(auc)

    return folds, float(np.mean(bces)), float(np.mean(aucs))

# ----------------------------
# 3) Permutation Importance (가중 BCE 증가량, n_repeats)
# ----------------------------
def permutation_importance(folds, feature_list, n_repeats=5, seed=42, pos_weight=POS_WEIGHT):
    rng = np.random.default_rng(seed)
    d = len(feature_list)
    deltas = np.zeros(d, dtype=np.float64)

    for j in range(d):
        incs = []
        for f in folds:
            base = f["loss_va"]
            for _ in range(n_repeats):
                Xv = f["X_va"].copy()
                rng.shuffle(Xv[:, j])  # j번째 피처만 섞기
                pv = sigmoid(Xv @ f["w"] + f["b"])
                Lp = bce_loss_weighted(f["y_va"], pv, pos_weight=pos_weight)
                incs.append(Lp - base)
        deltas[j] = np.mean(incs)

    # 오름차순: 덜 중요 → 중요
    return pd.Series(deltas, index=feature_list).sort_values(ascending=True)

# ----------------------------
# 4) 한 라운드: len//2개의 '덜 중요' 후보 중
#    각 1개씩만 제거 → 9-Fold 평가 → BCE 최저 (동률시 AUC 최대) 선택
# ----------------------------
def one_round(feature_list, base_epochs=1000, lr=0.05, l2=1e-4, pos_weight=POS_WEIGHT, tol=1e-4, verbose=True):
    folds, base_loss, base_auc = kfold_train_models(feature_list, epochs=base_epochs, lr=lr, l2=l2, pos_weight=pos_weight)

    imp = permutation_importance(folds, feature_list, n_repeats=5, seed=42, pos_weight=pos_weight)
    k = max(1, len(feature_list) // 2)
    worst_feats = imp.index[:k].tolist()

    scored = []
    for w in worst_feats:
        cand = [c for c in feature_list if c != w]
        _, loss_c, auc_c = kfold_train_models(cand, epochs=base_epochs, lr=lr, l2=l2, pos_weight=pos_weight)
        scored.append((loss_c, -auc_c, w, cand))  # BCE 낮을수록, AUC 높을수록 우선

    scored.sort(key=lambda x: (x[0], x[1]))
    best_loss, _, removed, best_cand = scored[0]

    if verbose:
        print(f"[ROUND] base_bce={base_loss:.6f} auc={base_auc:.4f} | best_bce={best_loss:.6f} | removed='{removed}' | feats={len(best_cand)}")

    improved = best_loss < (base_loss - tol)
    return best_cand, best_loss, improved, {"base_bce": base_loss, "base_auc": base_auc, "importance": imp, "scored": scored}

# ----------------------------
# 5) 반복 루프
# ----------------------------
def iterative_feature_prune(init_features=None, max_rounds=999, base_epochs=1000, lr=0.05, l2=1e-4, pos_weight=POS_WEIGHT):
    feats = list(init_features if init_features is not None else ALL_FEATURES)
    history = []

    for r in range(1, max_rounds + 1):
        if len(feats) <= 1:
            break
        print(f"\n=== Round {r} | #features={len(feats)} ===")
        feats_new, best_loss, improved, info = one_round(
            feats, base_epochs=base_epochs, lr=lr, l2=l2, pos_weight=pos_weight, tol=1e-4, verbose=True
        )
        history.append({"round": r, "best_bce": best_loss, "removed": list(set(feats) - set(feats_new)), "kept": feats_new})
        if not improved:
            print("No improvement → stop.")
            break
        feats = feats_new

    return feats, history

# ----------------------------
# 6) 실행
# ----------------------------
BEST_FEATURES, HIST = iterative_feature_prune(
    init_features=ALL_FEATURES,
    max_rounds=50,    # 필요시 ↑
    base_epochs=1000,  # 필요시 600~1000 ↑
    lr=0.05,
    l2=1e-4,          # L2 살짝 주면 BCE 안정화에 도움
    pos_weight=POS_WEIGHT
)

print("\n[RESULT]")
print("최종 피처 수 :", len(BEST_FEATURES))
print("최종 피처들 :", BEST_FEATURES)
print("라운드 수    :", len(HIST))
