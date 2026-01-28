# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

# =========================
# Config
# =========================
st.set_page_config(page_title="Steel Faults — HITL Dashboard", layout="wide")

NAVY = "#1800ad"
RED = "#e11d48"
LIGHT_BG = "rgba(255,255,255,0.06)"

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.0rem; }}
.card {{
  background: {LIGHT_BG};
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.18);
}}
.small {{ opacity: 0.92; font-size: 0.92rem; }}
hr {{ border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 10px 0; }}
.badge {{
  display:inline-block; padding:3px 9px; border-radius:999px;
  background: rgba(24,0,173,0.14); border:1px solid rgba(24,0,173,0.25);
  color: rgba(255,255,255,0.95); font-size: 0.85rem;
}}
</style>
""", unsafe_allow_html=True)

id2label = {
    0: "Other_Faults",
    1: "Bumps",
    2: "K_Scratch",
    3: "Z_Scratch",
    4: "Pastry",
    5: "Stains",
    6: "Dirtiness",
}
labels_sorted = sorted(id2label.keys())
label_list = [id2label[i] for i in labels_sorted]

PROCESS_HINTS = {
    "Other_Faults": "점검 포인트: 다른 결함들과 특징이 겹치기 쉬움. 상위 후보(top1/top2)와 현장 사진/로그로 교차 확인 권장.",
    "Bumps": "점검 포인트: 롤 표면/가이드 이물, 스케일(산화물) 눌림(rolled-in) 가능 구간, 압연 중 이물 부착 여부 확인.",
    "K_Scratch": "점검 포인트: 가이드/롤러 정렬, 롤 표면 거칠기/흠집, 하부면 접촉(가이드·구동롤) 긁힘 가능 구간 점검.",
    "Z_Scratch": "점검 포인트: 코일 층간 마찰, 상부면 성형·피드 롤 접촉, 이송 중 마찰/가이드 접촉 위치 확인.",
    "Pastry": "점검 포인트: 패치형 결함(눌림/국부 표면손상) 가능. 스케일/세척/산세 구간과 국부 표면 손상 조건 확인.",
    "Stains": "점검 포인트: 산세/세척/건조 라인, 오일·에멀전 잔사, 정지흔(stop mark) 등 ‘얼룩’ 유발 조건 점검.",
    "Dirtiness": "점검 포인트: 분진/파티클 유입, 세척/필터 상태, 작업장 청정도 및 롤/벨트 이물 부착 여부 확인.",
}

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    # 같은 폴더에 X_train.csv, y_train.csv, X_test.csv, y_test.csv 가 있다고 가정
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0].astype(int).to_numpy()
    X_test  = pd.read_csv("X_test.csv")
    y_test  = pd.read_csv("y_test.csv").iloc[:, 0].astype(int).to_numpy()
    return X_train, y_train, X_test, y_test

# =========================
# Feature Engineering
# =========================
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    eps = 1e-9

    # 안전하게: 컬럼이 없으면 해당 FE는 건너뜀
    if {"Pixels_Areas","Log_X_Index","Log_Y_Index"}.issubset(X.columns):
        X["shape_ratio"] = X["Pixels_Areas"] / ((X["Log_X_Index"] * X["Log_Y_Index"]) + eps)
        X["log_shape_ratio"] = np.log1p(X["shape_ratio"])

    if {"X_Perimeter","Y_Perimeter"}.issubset(X.columns):
        X["perimeter_sum"] = X["X_Perimeter"] + X["Y_Perimeter"] + eps

    if {"Pixels_Areas","perimeter_sum"}.issubset(X.columns):
        X["log_area_perimeter"] = np.log1p(X["Pixels_Areas"] / (X["perimeter_sum"] + eps))

    return X

# =========================
# Train model
# =========================
@st.cache_resource
def train_model(X_train_fe: pd.DataFrame, y_train: np.ndarray):
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=8,
        random_seed=42,
        verbose=0,
        use_best_model=False
    )
    model.fit(X_train_fe, y_train)
    return model

def predict_pack(model, X_fe: pd.DataFrame, thr: float):
    proba = model.predict_proba(X_fe)
    maxp = proba.max(axis=1)
    pred = proba.argmax(axis=1).astype(int)

    pred_final = pred.copy()
    pred_final[maxp < thr] = -1

    # top1/top2 and gap
    order = np.argsort(proba, axis=1)
    top1_id = order[:, -1]
    top2_id = order[:, -2]
    top1_p = proba[np.arange(len(proba)), top1_id]
    top2_p = proba[np.arange(len(proba)), top2_id]
    gap = top1_p - top2_p

    # entropy
    eps = 1e-12
    entropy = -(proba * np.log(proba + eps)).sum(axis=1)

    return dict(
        proba=proba, maxp=maxp, pred=pred, pred_final=pred_final,
        top1_id=top1_id, top2_id=top2_id, top1_p=top1_p, top2_p=top2_p,
        gap=gap, entropy=entropy
    )

def to_label(x):
    if int(x) == -1:
        return "REJECT"
    return id2label.get(int(x), "UNK")

def build_df(X_raw: pd.DataFrame, y_true: np.ndarray, pack: dict, split="TEST"):
    df = X_raw.copy()
    df["split"] = split
    df["sample_id"] = np.arange(len(df))  # 안정적 선택키

    df["y_true"] = y_true
    df["y_true_name"] = pd.Series(y_true).map(id2label)

    df["pred_id"] = pack["pred"]
    df["pred_name"] = pd.Series(pack["pred"]).map(id2label)

    df["pred_final_id"] = pack["pred_final"]
    df["pred_final_name"] = pd.Series(pack["pred_final"]).map(lambda v: to_label(v))

    df["maxp"] = pack["maxp"]
    df["top1_id"] = pack["top1_id"]
    df["top2_id"] = pack["top2_id"]
    df["top1_name"] = pd.Series(pack["top1_id"]).map(id2label)
    df["top2_name"] = pd.Series(pack["top2_id"]).map(id2label)
    df["top1_p"] = pack["top1_p"]
    df["top2_p"] = pack["top2_p"]
    df["gap"] = pack["gap"]
    df["entropy"] = pack["entropy"]
    df["is_reject"] = (df["pred_final_id"] == -1)

    # 좌표 중심(있으면)
    needed = {"X_Minimum","X_Maximum","Y_Minimum","Y_Maximum"}
    if needed.issubset(df.columns):
        df["Cx"] = (df["X_Minimum"] + df["X_Maximum"]) / 2
        df["Cy"] = (df["Y_Minimum"] + df["Y_Maximum"]) / 2

    # 강종 표시(있으면)
    if "TypeOfSteel_A300" in df.columns:
        df["steel_type"] = np.where(df["TypeOfSteel_A300"].astype(int) == 1, "A300", "A400")
    else:
        df["steel_type"] = "UNK"

    # 검수 우선순위(실무형): maxp 낮고, entropy 높고, gap 작을수록 ↑
    df["priority"] = (1.0 - df["maxp"]) + (df["entropy"] / (np.log(len(labels_sorted)) + 1e-9)) + (1.0 - df["gap"])
    return df

# =========================
# Small helpers (plots)
# =========================
def plot_maxp_hist(df: pd.DataFrame, thr: float, title: str):
    left = float((df["maxp"] < thr).mean() * 100)
    right = 100 - left

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["maxp"],
        nbinsx=45,
        marker=dict(color=NAVY),
        opacity=0.95
    ))
    fig.add_vline(x=thr, line_width=3, line_dash="dash", line_color=RED)
    fig.update_layout(
        title=f"{title} | max_proba 분포 (thr={thr:.2f})",
        xaxis_title="max_proba",
        yaxis_title="count",
        bargap=0.02,
        height=360,
        margin=dict(l=10,r=10,t=55,b=10),
    )
    fig.add_annotation(x=thr-0.02, y=1.03, xref="x", yref="paper",
                       text=f"<b>{left:.1f}%</b> (REJECT side)", showarrow=False, font=dict(color=RED))
    fig.add_annotation(x=thr+0.02, y=1.03, xref="x", yref="paper",
                       text=f"<b>{right:.1f}%</b> (AUTO side)", showarrow=False, font=dict(color=NAVY))
    return fig

def bar_class_reject_rate(df: pd.DataFrame, title: str):
    # true class 기준 reject rate
    g = df.groupby("y_true_name")["is_reject"].mean().sort_values(ascending=False)
    out = g.reset_index()
    out.columns = ["class", "reject_rate"]
    fig = px.bar(out, x="class", y="reject_rate", text=out["reject_rate"].map(lambda v: f"{v:.2f}"))
    fig.update_traces(marker_color=NAVY, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, yaxis_title="reject_rate", height=360, margin=dict(l=10,r=10,t=55,b=10))
    fig.update_yaxes(range=[0, min(1.0, float(out["reject_rate"].max()) * 1.25 + 0.02)])
    return fig

def bar_reject_top1(df: pd.DataFrame, title: str):
    rej = df[df["is_reject"]].copy()
    if len(rej) == 0:
        return None
    out = rej["top1_name"].value_counts().reset_index()
    out.columns = ["candidate(top1)", "count"]
    fig = px.bar(out, x="candidate(top1)", y="count", text="count")
    fig.update_traces(marker_color=NAVY, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, height=360, margin=dict(l=10,r=10,t=55,b=10))
    return fig

def plate_scatter(df_points: pd.DataFrame, selected_id=None, height=420, title="Plate Map"):
    fig = go.Figure()
    if df_points is None or len(df_points) == 0 or ("Cx" not in df_points.columns) or ("Cy" not in df_points.columns):
        fig.add_annotation(text="좌표가 없거나 표시할 데이터가 없습니다.", showarrow=False)
        fig.update_layout(height=height, margin=dict(l=10,r=10,t=55,b=10))
        return fig

    # base points
    fig.add_trace(go.Scatter(
        x=df_points["Cx"], y=df_points["Cy"],
        mode="markers",
        marker=dict(size=6, color=NAVY, opacity=0.55),
        text=df_points["top1_name"],
        hovertemplate="(%{x:.1f}, %{y:.1f})<br>top1=%{text}<extra></extra>"
    ))

    # highlight selected
    if selected_id is not None and (df_points["sample_id"] == selected_id).any():
        s = df_points[df_points["sample_id"] == selected_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[s["Cx"]], y=[s["Cy"]],
            mode="markers",
            marker=dict(size=14, color=RED, opacity=0.95, symbol="circle-open", line=dict(width=3, color=RED)),
            hovertemplate=f"SELECTED<br>id={int(selected_id)}<br>top1={s['top1_name']}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10,r=10,t=55,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    return fig

def tradeoff_curve(df_base: pd.DataFrame, proba: np.ndarray, y_true: np.ndarray):
    thrs = np.arange(0.50, 0.96, 0.02)
    rows = []
    for t in thrs:
        maxp = proba.max(axis=1)
        pred = proba.argmax(axis=1).astype(int)
        keep = maxp >= t
        keep_rate = float(k_



