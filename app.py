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
        keep_rate = float(keep.mean())
        if keep.sum() < 30:
            macro_f1 = np.nan
        else:
            rep = classification_report(y_true[keep], pred[keep], labels=labels_sorted, output_dict=True, zero_division=0)
            macro_f1 = float(rep["macro avg"]["f1-score"])
        rows.append([t, keep_rate, macro_f1])
    out = pd.DataFrame(rows, columns=["thr", "keep_rate", "macro_f1_on_keep"])
    return out

# =========================
# Main UI
# =========================
st.title("Steel Surface Faults — HITL 운영형 Dashboard (Auto + REJECT)")

X_train, y_train, X_test, y_test = load_data()
X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

with st.sidebar:
    st.header("운영 설정")
    thr = st.slider("Reject threshold (top1 prob < thr → REJECT)", 0.50, 0.99, 0.85, 0.01)
    st.caption("thr↑ : 안전(오판↓) / 검수량↑  |  thr↓ : 자동화↑ / 위험↑")

    page = st.radio("페이지", ["Overview", "Reject Queue", "Root Cause Explorer"])

    st.markdown("---")
    st.markdown("**표시 데이터**")
    show_split = st.selectbox("Split", ["TEST", "TRAIN"], index=0)

model = train_model(X_train_fe, y_train)

pack_tr = predict_pack(model, X_train_fe, thr)
pack_te = predict_pack(model, X_test_fe, thr)

df_tr = build_df(X_train, y_train, pack_tr, "TRAIN")
df_te = build_df(X_test,  y_test,  pack_te, "TEST")
df = df_te if show_split == "TEST" else df_tr

# =========================
# Page: Overview
# =========================
if page == "Overview":
    st.subheader("운영 요약 (한 장에서 끝내기)")

    total = len(df)
    keep = int((~df["is_reject"]).sum())
    rej = int(df["is_reject"].sum())
    keep_rate = keep / total if total else 0.0
    rej_rate = rej / total if total else 0.0

    # keep 성능 (확정된 것만)
    if keep >= 30:
        rep = classification_report(
            df.loc[~df["is_reject"], "y_true"].astype(int).to_numpy(),
            df.loc[~df["is_reject"], "pred_id"].astype(int).to_numpy(),
            labels=labels_sorted,
            target_names=label_list,
            digits=4,
            output_dict=True,
            zero_division=0
        )
        macro_f1 = float(rep["macro avg"]["f1-score"])
        acc = float(rep["accuracy"])
    else:
        macro_f1 = np.nan
        acc = np.nan

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{show_split} 총 샘플", f"{total:,}")
    c2.metric("자동확정(Keep)", f"{keep:,}", f"{keep_rate:.1%}")
    c3.metric("REJECT(검수)", f"{rej:,}", f"{rej_rate:.1%}")
    c4.metric("확정 샘플 성능", f"Acc {acc:.3f}" if acc==acc else "N/A", f"MacroF1 {macro_f1:.3f}" if macro_f1==macro_f1 else "")

    colA, colB = st.columns([0.58, 0.42], gap="large")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_maxp_hist(df, thr, title=f"{show_split}"), use_container_width=True)
        st.markdown('<div class="small">해석: thr(빨간선) 왼쪽은 <b>사람 검수</b>, 오른쪽은 <b>자동 확정</b> 영역.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = bar_class_reject_rate(df, title=f"{show_split}: 클래스별 REJECT율 (TRUE 기준)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="small">REJECT율이 높은 결함은 <b>피처 겹침/불확실성</b>이 큰 구간일 가능성이 큼.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    colC, colD = st.columns([0.58, 0.42], gap="large")
    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2 = bar_reject_top1(df, title=f"{show_split}: REJECT의 top1 후보 분포")
        if fig2 is None:
            st.info("현재 threshold에서 REJECT가 없습니다.")
        else:
            st.plotly_chart(fig2, use_container_width=True)
            top_fault = df[df["is_reject"]]["top1_name"].value_counts().index[0]
            st.warning(f"REJECT에서 가장 유력한 후보는 **{top_fault}**")
            st.caption(PROCESS_HINTS.get(top_fault, "점검 포인트 정보가 없습니다."))
        st.markdown('</div>', unsafe_allow_html=True)

    with colD:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### thr 트레이드오프 (자동화율 vs 확정성능)")
        st.caption("실무에서 ‘얼마나 자동화할지’ vs ‘오판 리스크’를 설명하는 핵심 그래프")

        base_proba = pack_te["proba"] if show_split == "TEST" else pack_tr["proba"]
        base_y = y_test if show_split == "TEST" else y_train

        curve = tradeoff_curve(df, base_proba, base_y)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=curve["thr"], y=curve["keep_rate"], mode="lines+markers", name="keep_rate", line=dict(color=NAVY)))
        fig3.add_trace(go.Scatter(x=curve["thr"], y=curve["macro_f1_on_keep"], mode="lines+markers", name="macro_f1_on_keep", line=dict(color="rgba(24,0,173,0.35)")))
        fig3.add_vline(x=thr, line_width=3, line_dash="dash", line_color=RED)
        fig3.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="thr", yaxis_title="value")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Page: Reject Queue
# =========================
elif page == "Reject Queue":
    st.subheader("REJECT 검수 큐 (안정형 UI: 필터 → 우선순위 → 상세 확인)")

    rej = df[df["is_reject"]].copy()
    if len(rej) == 0:
        st.success("현재 threshold에서 REJECT가 없습니다. (thr를 올리면 REJECT가 늘어납니다)")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns([0.28, 0.22, 0.22, 0.28])
        with f1:
            cand = st.multiselect("top1 후보(필터)", options=sorted(rej["top1_name"].unique().tolist()),
                                  default=sorted(rej["top1_name"].value_counts().head(3).index.tolist()))
        with f2:
            steel_opts = sorted(rej["steel_type"].unique().tolist())
            steel = st.multiselect("강종", options=steel_opts, default=steel_opts)
        with f3:
            sort_rule = st.selectbox("정렬", ["priority(추천)", "maxp↑(유력)", "gap↓(헷갈림)", "entropy↑(불확실)"])
        with f4:
            n_show = st.slider("표시 개수", 20, 300, 120, 10)

        sub = rej.copy()
        if cand:
            sub = sub[sub["top1_name"].isin(cand)]
        if steel:
            sub = sub[sub["steel_type"].isin(steel)]

        # 두께 필터(있으면)
        if "Steel_Plate_Thickness" in sub.columns and sub["Steel_Plate_Thickness"].notna().any():
            tmin, tmax = float(sub["Steel_Plate_Thickness"].min()), float(sub["Steel_Plate_Thickness"].max())
            t_lo, t_hi = st.slider("두께 범위", min_value=tmin, max_value=tmax, value=(tmin, tmax))
            sub = sub[(sub["Steel_Plate_Thickness"] >= t_lo) & (sub["Steel_Plate_Thickness"] <= t_hi)]

        # maxp 필터
        p_lo, p_hi = st.slider("maxp 범위", 0.0, 1.0, (0.0, float(thr)+0.10 if thr <= 0.89 else 1.0), 0.01)
        sub = sub[(sub["maxp"] >= p_lo) & (sub["maxp"] <= p_hi)]

        # sort
        if sort_rule == "priority(추천)":
            sub = sub.sort_values("priority", ascending=False)
        elif sort_rule == "maxp↑(유력)":
            sub = sub.sort_values("top1_p", ascending=False)
        elif sort_rule == "gap↓(헷갈림)":
            sub = sub.sort_values("gap", ascending=True)
        else:
            sub = sub.sort_values("entropy", ascending=False)

        st.markdown(f"<span class='badge'>Filtered REJECT: {len(sub):,} / {len(rej):,}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        left, right = st.columns([0.62, 0.38], gap="large")

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### REJECT 목록 (검수 우선순위)")
            view_cols = [
                "sample_id","steel_type","y_true_name","top1_name","top1_p","top2_name","top2_p",
                "maxp","gap","entropy","priority"
            ]
            view = sub[view_cols].head(n_show).copy()
            # 보기 좋게 라운딩
            for c in ["top1_p","top2_p","maxp","gap","entropy","priority"]:
                if c in view.columns:
                    view[c] = view[c].astype(float).round(4)
            st.dataframe(view, use_container_width=True, height=420)

            # 다운로드
            csv = sub[view_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button("REJECT 리스트 CSV 다운로드", data=csv, file_name="reject_queue.csv", mime="text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 선택 샘플 상세 (안정형)")
            # Streamlit row-click은 버전/환경마다 흔들리므로 selectbox로 안정 선택
            choices = sub["sample_id"].head(n_show).tolist()
            pick = st.selectbox("sample_id 선택", options=choices, index=0)

            row = sub[sub["sample_id"] == pick].iloc[0]

            st.write(f"**TRUE:** {row['y_true_name']}  |  **Top1:** {row['top1_name']} ({row['top1_p']:.3f})  |  **Top2:** {row['top2_name']} ({row['top2_p']:.3f})")
            st.write(f"**maxp:** {row['maxp']:.3f}  |  **gap(top1-top2):** {row['gap']:.3f}  |  **entropy:** {row['entropy']:.3f}")
            st.write(f"**steel:** {row['steel_type']}  |  **priority:** {row['priority']:.3f}")

            hint = PROCESS_HINTS.get(row["top1_name"], "점검 포인트 정보가 없습니다.")
            st.info(hint)

            # 좌표 맵(있으면)
            if "Cx" in sub.columns and "Cy" in sub.columns:
                st.plotly_chart(
                    plate_scatter(sub.head(2000), selected_id=pick, title="REJECT Map (top1 후보 필터 반영)"),
                    use_container_width=True
                )
            else:
                st.caption("좌표 컬럼(X/Y Min/Max)이 없어 맵을 표시할 수 없습니다.")

            st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Page: Root Cause Explorer
# =========================
else:
    st.subheader("Root Cause Explorer (강종/두께/피처 겹침으로 REJECT 원인 찾기)")

    # 1) 강종 x 결함 : reject_rate heatmap
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### (1) 강종(A300/A400) × 결함별 REJECT율 (TRUE 기준)")
    if "steel_type" in df.columns and df["steel_type"].nunique() > 1:
        pivot = df.pivot_table(index="y_true_name", columns="steel_type", values="is_reject", aggfunc="mean").fillna(0)
        fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("해석: 특정 강종에서 특정 결함의 REJECT율이 높으면 ‘피처 겹침/데이터 품질/공정 조건’ 가능성을 우선 점검.")
    else:
        st.info("TypeOfSteel_A300 컬럼이 없거나 강종 구분이 어려워 heatmap을 만들 수 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2) 두께 bin x 결함 : reject_rate
    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("#### (2) 두께 구간(bin) × 결함별 REJECT율 (TRUE 기준)")
    if "Steel_Plate_Thickness" in df.columns and df["Steel_Plate_Thickness"].notna().any():
        bins = st.slider("두께 bin 개수", 3, 12, 6, 1)
        tmp = df.copy()
        tmp["thk_bin"] = pd.qcut(tmp["Steel_Plate_Thickness"], q=bins, duplicates="drop")
        pivot2 = tmp.pivot_table(index="y_true_name", columns="thk_bin", values="is_reject", aggfunc="mean").fillna(0)
        fig2 = px.imshow(pivot2, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig2.update_layout(height=420, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("해석: 두께 구간에 따라 특정 결함 REJECT율이 높아지면, 해당 두께 조건에서 결함 특성이 겹치거나 센서/조명 조건이 달라질 가능성.")
    else:
        st.info("Steel_Plate_Thickness 컬럼이 없어 두께 기반 분석을 건너뜁니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3) 2D feature overlap (confusion)
    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("#### (3) 피처 공간에서 ‘겹침’ 확인 (REJECT 강조)")

    # 선택 결함
    c_name = st.selectbox("TRUE 결함 선택", options=label_list, index=label_list.index("Pastry") if "Pastry" in label_list else 0)
    c_id = {v:k for k,v in id2label.items()}[c_name]

    # 가장 많이 헷갈리는 상대 클래스(Top1 후보) 자동 선택
    tmp_rej = df[(df["is_reject"]) & (df["y_true"] == c_id)]
    if len(tmp_rej) > 0:
        other_id = int(tmp_rej["top1_id"].value_counts().index[0])
    else:
        other_id = c_id

    other_name = id2label.get(other_id, "UNK")

    # 2D feature 선택
    candidates = [c for c in X_test_fe.columns if c not in ["y_true","target"]]
    default_x = "Pixels_Areas" if "Pixels_Areas" in candidates else candidates[0]
    default_y = "perimeter_sum" if "perimeter_sum" in candidates else candidates[min(1, len(candidates)-1)]

    fx = st.selectbox("X-axis feature", options=candidates, index=candidates.index(default_x))
    fy = st.selectbox("Y-axis feature", options=candidates, index=candidates.index(default_y))

    # build plot df
    feat_df = add_features(df.drop(columns=["split","sample_id"], errors="ignore"))  # df에는 원래 raw가 있으니 FE 다시 맞춰줌
    # 하지만 df는 raw X + 메타가 섞여있어서, FE는 raw 컬럼이 있을 때만 동작
    # 여기서는 df에서 바로 fx, fy를 읽을 수 있어야 하므로, 없으면 X_test_fe에서 이어붙임
    if fx not in df.columns or fy not in df.columns:
        # 안전하게: df에 없는 feature는 split에 맞춰 X_fe에서 가져오기
        X_fe = X_test_fe if df["split"].iloc[0] == "TEST" else X_train_fe
        df2 = df.copy()
        for col in [fx, fy]:
            if col in X_fe.columns:
                df2[col] = X_fe[col].values
    else:
        df2 = df.copy()

    keep = df2["y_true_name"].isin([c_name, other_name])
    sub = df2[keep].copy()
    sub["group"] = np.where(sub["y_true_name"] == c_name, c_name, other_name)
    sub["reject_tag"] = np.where(sub["is_reject"], "REJECT", "OK")

    title = f"{c_name} vs {other_name} | 2D overlap ({fx} vs {fy}) — REJECT 강조"
    fig3 = px.scatter(
        sub, x=fx, y=fy,
        color="group",
        symbol="reject_tag",
        opacity=0.65,
        title=title
    )
    fig3.update_traces(marker=dict(size=7))
    fig3.update_layout(height=520, margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("해석: REJECT 포인트가 두 결함의 경계/겹침 구간에 몰리면 ‘피처적으로 구분이 어려운 영역’이 존재한다는 강한 근거.")
    st.markdown('</div>', unsafe_allow_html=True)




