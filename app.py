
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
st.set_page_config(page_title="Steel Faults Dashboard", layout="wide")

# ---- CSS (추가 꾸밈: 카드/간격/폰트) ----
st.markdown("""
<style>
/* 카드 느낌 */
.block-container { padding-top: 1.2rem; }
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.22);
}
.small { opacity: 0.92; font-size: 0.92rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 10px 0; }
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

# =========================
# Process hints (간단 버전)
# =========================
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
    X["shape_ratio"] = X["Pixels_Areas"] / ((X["Log_X_Index"] * X["Log_Y_Index"]) + eps)
    X["log_shape_ratio"] = np.log1p(X["shape_ratio"])
    X["perimeter_sum"] = X["X_Perimeter"] + X["Y_Perimeter"] + eps
    X["log_area_perimeter"] = np.log1p(X["Pixels_Areas"] / X["perimeter_sum"])
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

def predict_with_reject(model, X_fe: pd.DataFrame, thr: float):
    proba = model.predict_proba(X_fe)
    maxp = proba.max(axis=1)
    pred = proba.argmax(axis=1).astype(int)

    pred_final = pred.copy()
    pred_final[maxp < thr] = -1

    proba_sorted = np.sort(proba, axis=1)[:, ::-1]
    top1 = proba_sorted[:, 0]
    top2 = proba_sorted[:, 1]
    gap = top1 - top2

    eps = 1e-12
    entropy = -(proba * np.log(proba + eps)).sum(axis=1)

    top1_id = proba.argmax(axis=1)
    top2_id = np.argsort(proba, axis=1)[:, -2]

    return {
        "proba": proba,
        "maxp": maxp,
        "pred": pred,
        "pred_final": pred_final,
        "top1_p": top1,
        "top2_p": top2,
        "gap": gap,
        "entropy": entropy,
        "top1_id": top1_id,
        "top2_id": top2_id,
    }

def to_label(arr):
    return pd.Series(arr).map(lambda x: id2label.get(int(x), "REJECT") if x != -1 else "REJECT")

def build_result_df(X_raw, y_true, pred_pack, split_name="TEST"):
    df = X_raw.copy()
    df["split"] = split_name
    df["y_true"] = y_true
    df["y_true_name"] = pd.Series(y_true).map(id2label)

    df["pred_id"] = pred_pack["pred"]
    df["pred_name"] = to_label(pred_pack["pred"])
    df["pred_final_id"] = pred_pack["pred_final"]
    df["pred_final_name"] = to_label(pred_pack["pred_final"])

    df["maxp"] = pred_pack["maxp"]
    df["top1_id"] = pred_pack["top1_id"]
    df["top2_id"] = pred_pack["top2_id"]
    df["top1_name"] = pd.Series(pred_pack["top1_id"]).map(id2label)
    df["top2_name"] = pd.Series(pred_pack["top2_id"]).map(id2label)
    df["top1_p"] = pred_pack["top1_p"]
    df["top2_p"] = pred_pack["top2_p"]
    df["gap"] = pred_pack["gap"]
    df["entropy"] = pred_pack["entropy"]
    df["is_reject"] = (df["pred_final_id"] == -1)

    # 좌표 중심(있으면)
    needed = {"X_Minimum","X_Maximum","Y_Minimum","Y_Maximum"}
    if needed.issubset(df.columns):
        df["Cx"] = (df["X_Minimum"] + df["X_Maximum"]) / 2
        df["Cy"] = (df["Y_Minimum"] + df["Y_Maximum"]) / 2

    return df

# =========================
# Plate plot + mini slices
# =========================
def plate_scatter(df_points: pd.DataFrame, title: str, height: int = 430):
    fig = go.Figure()
    if df_points is None or len(df_points) == 0 or ("Cx" not in df_points.columns) or ("Cy" not in df_points.columns):
        fig.add_annotation(text="표시할 좌표가 없습니다.", showarrow=False)
        fig.update_layout(height=height, margin=dict(l=10,r=10,t=55,b=10))
        return fig

    x = df_points["Cx"].values
    y = df_points["Cy"].values
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))

    fig.add_shape(
        type="rect",
        x0=xmin, y0=ymin, x1=xmax, y1=ymax,
        line=dict(width=2, color="rgba(170,210,255,0.35)"),
        fillcolor="rgba(255,255,255,0.05)"
    )

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=6, color="#FF3B30", opacity=0.85),
        text=df_points.get("pred_final_name", None),
        hovertemplate="(%{x:.1f}, %{y:.1f})<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10,r=10,t=55,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def make_mini_slices(df_def: pd.DataFrame, slice_col: str = "Length_of_Conveyer", n_slices: int = 8):
    if df_def is None or len(df_def) == 0:
        return []
    if slice_col in df_def.columns and df_def[slice_col].notna().sum() > 10:
        bins = pd.qcut(df_def[slice_col], q=n_slices, duplicates="drop")
        groups = []
        for b in bins.cat.categories:
            part = df_def[bins == b]
            groups.append((f"{slice_col} {b}", part))
        return groups
    else:
        idx = np.array_split(df_def.index.to_numpy(), n_slices)
        groups = []
        for i, ids in enumerate(idx, start=1):
            part = df_def.loc[ids]
            groups.append((f"Slice {i}", part))
        return groups

# =========================
# UI
# =========================
st.title("Steel Plate Faults — Auto 판정 + Reject(사람 검수) Dashboard")

X_train, y_train, X_test, y_test = load_data()
X_train_fe = add_features(X_train)
X_test_fe  = add_features(X_test)

with st.sidebar:
    st.header("설정")
    thr = st.slider("Reject threshold (top1 prob < thr → REJECT)", 0.50, 0.99, 0.85, 0.01)
    st.caption("thr를 올리면 유출 위험은 줄지만 Reject(검수량)는 늘어남")
    page = st.radio("페이지", ["Overview", "Defect Cockpit", "Reject Triage", "Model Health"])

model = train_model(X_train_fe, y_train)

pred_train = predict_with_reject(model, X_train_fe, thr)
pred_test  = predict_with_reject(model, X_test_fe, thr)

df_tr = build_result_df(X_train, y_train, pred_train, "TRAIN")
df_te = build_result_df(X_test, y_test, pred_test, "TEST")

# =========================
# Overview
# =========================
if page == "Overview":
    st.subheader("운영 요약 (현재 threshold 기준)")

    c1, c2, c3, c4 = st.columns(4)
    for df, title, c in [(df_tr, "TRAIN", c1), (df_te, "TEST", c2)]:
        total = len(df)
        keep = int((~df["is_reject"]).sum())
        rej = int(df["is_reject"].sum())
        c.metric(f"{title} 총 검사", f"{total:,}")
        c.metric(f"{title} 확정(자동판정)", f"{keep:,}", f"{keep/total:.1%}")
        c.metric(f"{title} REJECT(검수)", f"{rej:,}", f"{rej/total:.1%}")

    st.markdown("### TEST: 자동 확정된 결함 분포")
    df_keep = df_te[~df_te["is_reject"]].copy()
    dist = df_keep["pred_final_name"].value_counts().reset_index()
    dist.columns = ["fault", "count"]
    fig = px.bar(dist, x="fault", y="count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### TEST: REJECT에서 '가장 유력한 후보 결함' TOP")
    rej = df_te[df_te["is_reject"]].copy()
    if len(rej) == 0:
        st.info("현재 threshold에서 REJECT가 없습니다.")
    else:
        top_fault = rej["top1_name"].value_counts().reset_index()
        top_fault.columns = ["top1_candidate_fault", "count"]
        fig2 = px.bar(top_fault, x="top1_candidate_fault", y="count")
        st.plotly_chart(fig2, use_container_width=True)

        top_rej_fault = top_fault.sort_values("count", ascending=False).iloc[0]["top1_candidate_fault"]
        st.warning(f"REJECT 중 가장 유력한 후보 결함은 **{top_rej_fault}** 입니다.")
        st.caption(PROCESS_HINTS.get(top_rej_fault, "점검 포인트 정보가 아직 등록되지 않았습니다."))

# =========================
# Defect Cockpit (요구한 화면)
# =========================
elif page == "Defect Cockpit":
    st.subheader("Defect Cockpit")

    if "Cx" not in df_te.columns or "Cy" not in df_te.columns:
        st.warning("좌표 컬럼(X/Y Min/Max)이 없어 Cockpit 맵을 그릴 수 없습니다.")
    else:
        df = df_te.copy()

        left, right = st.columns([0.36, 0.64], gap="large")

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 자동확정(≥thr) 결함 분포")
            df_keep = df[~df["is_reject"]].copy()
            if len(df_keep) == 0:
                st.info("현재 threshold에서 자동확정 샘플이 없습니다. thr를 낮춰보세요.")
            else:
                dist = df_keep["pred_final_name"].value_counts().reset_index()
                dist.columns = ["fault", "count"]
                fig = px.bar(dist, x="fault", y="count")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("#### REJECT: top1 후보 결함 분포")
            rej = df[df["is_reject"]].copy()
            if len(rej) == 0:
                st.success("현재 threshold에서 REJECT가 없습니다.")
            else:
                top_fault = rej["top1_name"].value_counts().reset_index()
                top_fault.columns = ["candidate_fault", "count"]
                fig2 = px.bar(top_fault, x="candidate_fault", y="count")
                st.plotly_chart(fig2, use_container_width=True)

                top_rej_fault = top_fault.sort_values("count", ascending=False).iloc[0]["candidate_fault"]
                st.warning(f"REJECT 중 가장 유력한 후보 결함은 **{top_rej_fault}** 입니다.")
                st.caption(PROCESS_HINTS.get(top_rej_fault, "점검 포인트 정보가 아직 등록되지 않았습니다."))
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 철판 결함 위치 맵 (빨간 점)")

            fault_list = ["ALL", "REJECT"] + [id2label[i] for i in labels_sorted]
            try:
                selected_fault = st.segmented_control("결함 선택", options=fault_list, default="ALL")
            except Exception:
                selected_fault = st.selectbox("결함 선택", fault_list, index=0)

            if selected_fault == "ALL":
                df_map = df.copy()
                title = "ALL (confirmed + reject)"
            elif selected_fault == "REJECT":
                df_map = df[df["is_reject"]].copy()
                title = "REJECT only"
            else:
                df_map = df[df["pred_final_name"] == selected_fault].copy()
                title = f"{selected_fault} only"

            st.plotly_chart(plate_scatter(df_map, title=title, height=430), use_container_width=True)

            if selected_fault in PROCESS_HINTS:
                st.info(PROCESS_HINTS[selected_fault])
            elif selected_fault == "REJECT":
                st.caption("REJECT는 (top1/top2, gap, entropy)를 함께 보고 검수 우선순위를 정하는 것이 좋습니다.")
            else:
                st.caption("결함을 선택하면 해당 결함의 점검 포인트 문구가 자동으로 표시됩니다.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("#### 미니 철판맵 슬라이드 (선택 결함)")
            st.caption("◀ ▶ 버튼으로 3장씩 넘겨보세요. (화면을 꽉 채우지 않도록 미니맵)")

            if selected_fault in ["ALL", "REJECT"]:
                st.caption("미니맵은 특정 결함을 선택했을 때만 표시합니다.")
            else:
                df_def = df[df["pred_final_name"] == selected_fault].copy()
                slices = make_mini_slices(df_def, slice_col="Length_of_Conveyer", n_slices=8)

                if len(slices) == 0:
                    st.warning("표시할 미니맵 데이터가 없습니다.")
                else:
                    key = f"slide_idx_{selected_fault}"
                    if key not in st.session_state:
                        st.session_state[key] = 0

                    a, b, c = st.columns([0.12, 0.12, 0.76])
                    with a:
                        if st.button("◀", use_container_width=True):
                            st.session_state[key] = max(0, st.session_state[key] - 3)
                    with b:
                        if st.button("▶", use_container_width=True):
                            st.session_state[key] = min(len(slices) - 1, st.session_state[key] + 3)
                    with c:
                        st.write(f"{st.session_state[key]+1} ~ {min(st.session_state[key]+3, len(slices))} / 총 {len(slices)}")

                    start = st.session_state[key]
                    chunk = slices[start:start+3]
                    cols = st.columns(len(chunk))
                    for i, (t, part) in enumerate(chunk):
                        with cols[i]:
                            st.plotly_chart(plate_scatter(part, title=t, height=250), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Reject Triage
# =========================
elif page == "Reject Triage":
    st.subheader("REJECT (사람 검수 우선순위)")

    rej = df_te[df_te["is_reject"]].copy()
    if len(rej) == 0:
        st.info("현재 threshold에서 REJECT가 없습니다.")
    else:
        sort_rule = st.selectbox(
            "정렬 기준",
            ["top1_p 높은 순(유력 후보)", "gap 작은 순(헷갈림)", "entropy 높은 순(불확실)", "top1_p & gap 혼합"]
        )

        if sort_rule == "top1_p 높은 순(유력 후보)":
            rej = rej.sort_values("top1_p", ascending=False)
        elif sort_rule == "gap 작은 순(헷갈림)":
            rej = rej.sort_values("gap", ascending=True)
        elif sort_rule == "entropy 높은 순(불확실)":
            rej = rej.sort_values("entropy", ascending=False)
        else:
            rej["priority_score"] = rej["top1_p"] - rej["gap"]
            rej = rej.sort_values("priority_score", ascending=False)

        st.markdown("### REJECT 중 ‘가장 높은 후보 결함(top1)’ + 확률")
        st.dataframe(
            rej[["top1_name","top1_p","top2_name","top2_p","gap","entropy","y_true_name"]].head(100),
            use_container_width=True
        )

        st.markdown("### REJECT 후보 결함 분포(top1 기준)")
        dist = rej["top1_name"].value_counts().reset_index()
        dist.columns = ["candidate_fault", "count"]
        fig = px.bar(dist, x="candidate_fault", y="count")
        st.plotly_chart(fig, use_container_width=True)

        out_cols = ["top1_name","top1_p","top2_name","top2_p","gap","entropy","y_true_name"]
        csv = rej[out_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("REJECT 리스트 CSV 다운로드", data=csv, file_name="reject_queue.csv", mime="text/csv")

# =========================
# Model Health
# =========================
else:
    st.subheader("Model Health (확정된 것만 기준 성능)")

    df_keep = df_te[~df_te["is_reject"]].copy()
    if len(df_keep) == 0:
        st.warning("확정된 샘플이 없습니다. threshold를 낮춰보세요.")
    else:
        y_true_keep = df_keep["y_true"].astype(int).to_numpy()
        y_pred_keep = df_keep["pred_final_id"].astype(int).to_numpy()

        report = classification_report(
            y_true_keep, y_pred_keep,
            labels=labels_sorted,
            target_names=[id2label[i] for i in labels_sorted],
            digits=4,
            output_dict=True
        )
        rep_df = pd.DataFrame(report).T.reset_index().rename(columns={"index":"metric"})
        st.dataframe(rep_df, use_container_width=True)

        cm = confusion_matrix(y_true_keep, y_pred_keep, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=[id2label[i] for i in labels_sorted], columns=[id2label[i] for i in labels_sorted])
        fig = px.imshow(cm_df, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 확정율/REJECT율 변화는 운영에서 중요한 드리프트 신호가 될 수 있습니다.")





