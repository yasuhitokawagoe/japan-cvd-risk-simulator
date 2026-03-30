import streamlit as st
import plotly.graph_objects as go
import numpy as np
from calc_engine_outcomes import OutcomesEngine

st.set_page_config(
    page_title="一次予防リスク（モバイル）",
    layout="centered",
    page_icon="🫀",
)

st.title("🫀 一次予防リスクシミュレーター（モバイル版）")
st.caption("将来の心血管リスクと、改善した場合の変化を簡単に確認できます。")

engine = OutcomesEngine("config.yaml")

if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.cumulative_data = None


def calculate_cumulative_curves():
    if which == "5-year":
        years = 5
    elif which == "10-year":
        years = 10
    elif which == "20-year":
        years = 20
    elif which == "30-year":
        years = 30
    elif which == "50-year":
        years = 50

    calc_years = np.arange(1, years + 1, 1)
    cumulative_data = {}

    for outcome in ["mi", "stroke", "mortality"]:
        cumulative_data[outcome] = {
            "baseline_cumulative": [],
            "target_cumulative": [],
            "baseline_ci_lower": [],
            "baseline_ci_upper": [],
            "target_ci_lower": [],
            "target_ci_upper": [],
        }
        cumulative_data[outcome]["time"] = [0.0]
        cumulative_data[outcome]["baseline_cumulative"] = [0.0]
        cumulative_data[outcome]["target_cumulative"] = [0.0]
        cumulative_data[outcome]["baseline_ci_lower"] = [0.0]
        cumulative_data[outcome]["baseline_ci_upper"] = [0.0]
        cumulative_data[outcome]["target_ci_lower"] = [0.0]
        cumulative_data[outcome]["target_ci_upper"] = [0.0]

        AGE_CAP = 110
        for y in calc_years:
            age_at_t = age + y
            if age_at_t > AGE_CAP:
                break

            res = engine.cumulative_incidence_with_ci(
                outcome,
                sex,
                age,
                int(y),
                sbp_now,
                sbp_tgt,
                ldl_now,
                ldl_tgt,
                a1c_now,
                a1c_tgt,
                smoking_status,
                cigs_per_day,
                years_smoked,
                years_since_quit,
                quit_today,
                bmi_now=None,
                bmi_target=None,
                egfr_now=None,
                egfr_target=None,
                acr_now=None,
                acr_target=None,
            )
            cumulative_data[outcome]["time"].append(float(y))
            cumulative_data[outcome]["baseline_cumulative"].append(res["point"]["baseline"] * 100.0)
            cumulative_data[outcome]["target_cumulative"].append(res["point"]["target"] * 100.0)
            cumulative_data[outcome]["baseline_ci_lower"].append(res["lower"]["baseline"] * 100.0)
            cumulative_data[outcome]["baseline_ci_upper"].append(res["upper"]["baseline"] * 100.0)
            cumulative_data[outcome]["target_ci_lower"].append(res["lower"]["target"] * 100.0)
            cumulative_data[outcome]["target_ci_upper"].append(res["upper"]["target"] * 100.0)

    from scipy.interpolate import make_interp_spline

    for outcome in ["mi", "stroke", "mortality"]:
        ts = np.array(cumulative_data[outcome]["time"], dtype=float)
        base = np.array(cumulative_data[outcome]["baseline_cumulative"], dtype=float)
        targ = np.array(cumulative_data[outcome]["target_cumulative"], dtype=float)
        bl_l = np.array(cumulative_data[outcome]["baseline_ci_lower"], dtype=float)
        bl_u = np.array(cumulative_data[outcome]["baseline_ci_upper"], dtype=float)
        tg_l = np.array(cumulative_data[outcome]["target_ci_lower"], dtype=float)
        tg_u = np.array(cumulative_data[outcome]["target_ci_upper"], dtype=float)

        if len(ts) >= 4:
            dense_times = np.linspace(ts[0], ts[-1], max(101, int((ts[-1] - ts[0]) * 20)))
            base_s = make_interp_spline(ts, base, k=3)(dense_times)
            targ_s = make_interp_spline(ts, targ, k=3)(dense_times)
            bl_l_s = np.interp(dense_times, ts, bl_l)
            bl_u_s = np.interp(dense_times, ts, bl_u)
            tg_l_s = np.interp(dense_times, ts, tg_l)
            tg_u_s = np.interp(dense_times, ts, tg_u)

            cumulative_data[outcome]["time"] = dense_times
            cumulative_data[outcome]["baseline_cumulative"] = base_s
            cumulative_data[outcome]["target_cumulative"] = targ_s
            cumulative_data[outcome]["baseline_ci_lower"] = bl_l_s
            cumulative_data[outcome]["baseline_ci_upper"] = bl_u_s
            cumulative_data[outcome]["target_ci_lower"] = tg_l_s
            cumulative_data[outcome]["target_ci_upper"] = tg_u_s

    return cumulative_data


def _smooth_main_lines(fig):
    for trace in fig.data:
        if (
            trace.mode == "lines"
            and hasattr(trace, "name")
            and trace.name
            and "95%CI" not in trace.name
        ):
            trace.update(line=dict(smoothing=1.0, shape="spline"))


def figure_mi(cumulative_data, age):
    fig = go.Figure()
    _t = np.array(cumulative_data["mi"]["time"], dtype=float)
    _b = np.array(cumulative_data["mi"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["mi"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name="現在のリスク因子",
            line=dict(color="#ff6b6b", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name="現在のリスク因子（≥85歳推定域）",
            line=dict(color="rgba(255,107,107,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_tg[:cut_idx],
            mode="lines",
            name="目標達成時",
            line=dict(color="#4ecdc4", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name="目標達成時（≥85歳推定域）",
            line=dict(color="rgba(78,205,196,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mi"]["time"],
            y=cumulative_data["mi"]["baseline_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mi"]["time"],
            y=cumulative_data["mi"]["baseline_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="現在のリスク因子 95%CI",
            fillcolor="rgba(255,107,107,0.2)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mi"]["time"],
            y=cumulative_data["mi"]["target_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mi"]["time"],
            y=cumulative_data["mi"]["target_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="目標達成時 95%CI",
            fillcolor="rgba(78,205,196,0.2)",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="心筋梗塞・累積リスク（%）",
        xaxis_title="年数",
        yaxis_title="累積リスク（%）",
        height=320,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=44, b=60),
    )
    _smooth_main_lines(fig)
    return fig


def figure_stroke(cumulative_data, age):
    fig = go.Figure()
    _t = np.array(cumulative_data["stroke"]["time"], dtype=float)
    _b = np.array(cumulative_data["stroke"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["stroke"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name="現在のリスク因子",
            line=dict(color="#ffa726", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name="現在のリスク因子（≥85歳推定域）",
            line=dict(color="rgba(255,167,38,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_tg[:cut_idx],
            mode="lines",
            name="目標達成時",
            line=dict(color="#66bb6a", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name="目標達成時（≥85歳推定域）",
            line=dict(color="rgba(102,187,106,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["stroke"]["time"],
            y=cumulative_data["stroke"]["baseline_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["stroke"]["time"],
            y=cumulative_data["stroke"]["baseline_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="現在のリスク因子 95%CI",
            fillcolor="rgba(255,167,38,0.2)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["stroke"]["time"],
            y=cumulative_data["stroke"]["target_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["stroke"]["time"],
            y=cumulative_data["stroke"]["target_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="目標達成時 95%CI",
            fillcolor="rgba(102,187,106,0.2)",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="脳卒中・累積リスク（%）",
        xaxis_title="年数",
        yaxis_title="累積リスク（%）",
        height=320,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=44, b=60),
    )
    _smooth_main_lines(fig)
    return fig


def figure_mortality(cumulative_data, age):
    fig = go.Figure()
    _t = np.array(cumulative_data["mortality"]["time"], dtype=float)
    _b = np.array(cumulative_data["mortality"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["mortality"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name="現在のリスク因子",
            line=dict(color="#ef5350", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name="現在のリスク因子（≥85歳推定域）",
            line=dict(color="rgba(239,83,80,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_tg[:cut_idx],
            mode="lines",
            name="目標達成時",
            line=dict(color="#26a69a", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name="目標達成時（≥85歳推定域）",
            line=dict(color="rgba(38,166,154,0.45)", width=2),
            showlegend=False,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mortality"]["time"],
            y=cumulative_data["mortality"]["baseline_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mortality"]["time"],
            y=cumulative_data["mortality"]["baseline_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="現在のリスク因子 95%CI",
            fillcolor="rgba(239,83,80,0.2)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mortality"]["time"],
            y=cumulative_data["mortality"]["target_ci_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_data["mortality"]["time"],
            y=cumulative_data["mortality"]["target_ci_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="目標達成時 95%CI",
            fillcolor="rgba(38,166,154,0.2)",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="全死亡・累積リスク（%）",
        xaxis_title="年数",
        yaxis_title="累積リスク（%）",
        height=320,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=44, b=60),
    )
    _smooth_main_lines(fig)
    return fig


st.subheader("入力")
sex = st.selectbox(
    "性別",
    ["male", "female"],
    format_func=lambda x: "男性" if x == "male" else "女性",
)
age = st.number_input("年齢（歳）", 20, 95, 60, step=1)

sbp_now = st.slider("収縮期血圧 現在 (mmHg)", 90, 200, 150)
sbp_tgt = st.slider("収縮期血圧 目標 (mmHg)", 90, 160, 130)

ldl_now = st.slider("LDLコレステロール 現在 (mg/dL)", 50, 250, 160)
ldl_tgt = st.slider("LDLコレステロール 目標 (mg/dL)", 50, 160, 100)

a1c_now = st.slider("HbA1c 現在 (%)", 5.0, 12.0, 8.0, step=0.1)
a1c_tgt = st.slider("HbA1c 目標 (%)", 5.0, 9.0, 7.0, step=0.1)

smoking_status = st.selectbox(
    "喫煙状況",
    ["never", "current", "former"],
    format_func=lambda x: {"never": "非喫煙者", "current": "現在喫煙者", "former": "元喫煙者"}[x],
)
if smoking_status == "never":
    cigs_per_day = 0
    years_smoked = 0.0
    years_since_quit = 0.0
    quit_today = False
elif smoking_status == "current":
    cigs_per_day = st.slider("1日あたりの喫煙本数", 0, 40, 20)
    years_smoked = st.slider("喫煙年数", 0, 60, 20)
    years_since_quit = 0.0
    quit_today = st.checkbox("今日禁煙したと仮定（目標シナリオ）")
else:
    cigs_per_day = st.slider("1日あたりの喫煙本数", 0, 40, 20)
    years_smoked = st.slider("喫煙年数", 0, 60, 20)
    years_since_quit = st.slider("禁煙からの年数（元喫煙者の場合）", 0, 40, 5)
    quit_today = False

which = st.radio(
    "予測期間",
    ["5-year", "10-year", "20-year", "30-year", "50-year"],
    index=2,
    format_func=lambda x: {
        "5-year": "5年",
        "10-year": "10年",
        "20-year": "20年",
        "30-year": "30年",
        "50-year": "50年",
    }[x],
)

if st.button("🔄 リスク計算を実行", type="primary"):
    st.session_state.cumulative_data = calculate_cumulative_curves()
    st.session_state.calculated = True

if st.session_state.calculated and st.session_state.cumulative_data is not None:
    cumulative_data = st.session_state.cumulative_data
else:
    st.info("入力のあと「🔄 リスク計算を実行」を押してください。")
    st.stop()

if which == "5-year":
    horizons = [5]
elif which == "10-year":
    horizons = [10]
elif which == "20-year":
    horizons = [20]
elif which == "30-year":
    horizons = [30]
else:
    horizons = [50]

h = horizons[0]
labels = {"mi": "心筋梗塞", "stroke": "脳卒中", "mortality": "全死亡"}

r_by_outcome = {}
for outcome in ["mi", "stroke", "mortality"]:
    r_by_outcome[outcome] = engine.cumulative_incidence(
        outcome,
        sex,
        age,
        h,
        sbp_now,
        sbp_tgt,
        ldl_now,
        ldl_tgt,
        a1c_now,
        a1c_tgt,
        smoking_status,
        cigs_per_day,
        years_smoked,
        years_since_quit,
        assume_quit_today_in_target=quit_today,
    )

st.markdown("#### 結果サマリー")
for outcome in ["mi", "stroke", "mortality"]:
    r = r_by_outcome[outcome]
    diff = r["baseline"] - r["target"]
    st.markdown(
        f"**{labels[outcome]}**（{h}年）: "
        f"現在 **{100 * r['baseline']:.1f}%** → 目標 **{100 * r['target']:.1f}%** · "
        f"差 **{100 * diff:+.1f}%**"
    )

st.divider()
st.markdown("### 詳細表示")

detail_blocks = [
    ("mi", "🫀 心筋梗塞", figure_mi),
    ("stroke", "🧠 脳卒中", figure_stroke),
    ("mortality", "💀 全死亡", figure_mortality),
]

DETAIL_GRAPH_CAPTION = (
    "🔴 現在のリスク因子　🟢 目標達成時　薄い帯：95%信頼区間　薄い線：85歳以上推定域"
)

for outcome_key, heading, fig_fn in detail_blocks:
    st.subheader(heading)
    fig = fig_fn(cumulative_data, age)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(DETAIL_GRAPH_CAPTION)

    r = r_by_outcome[outcome_key]
    diff = r["baseline"] - r["target"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("現在", f"{100 * r['baseline']:.1f}%")
    with c2:
        st.metric("目標", f"{100 * r['target']:.1f}%")
    with c3:
        st.metric("差（削減）", f"{100 * diff:.1f}%")
    st.markdown("---")

with st.expander("簡易注記"):
    st.markdown(
        """
- 教育・共有意思決定向けの簡易表示です。医療機器ではありません。
- 本画面は BMI・CKD を含みません（`app_streamlit_outcomes.py` の PC 版で入力できます）。
"""
    )
