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

    for outcome in ["mortality", "mi", "stroke"]:
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

    for outcome in ["mortality", "mi", "stroke"]:
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


def _bump(key, delta, lo, hi):
    """クイック調整ボタン用：session_state の値を増減し範囲内に丸める。UIのみ。"""
    new = st.session_state[key] + delta
    new = max(lo, min(hi, new))
    if isinstance(st.session_state[key], float):
        new = round(new, 1)
    st.session_state[key] = new


_input_defaults = {
    "sbp_now": 150,
    "sbp_tgt": 130,
    "ldl_now": 180,
    "ldl_tgt": 100,
    "a1c_now": 7.0,
    "a1c_tgt": 6.5,
    "bmi_now_ui": 29.0,
    "cigs_per_day": 20,
    "years_smoked": 20,
    "years_since_quit": 5,
}
for _k, _v in _input_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.subheader("入力")

with st.container(border=True):
    st.markdown("#### 基本情報")
    bi1, bi2, bi3 = st.columns(3)
    with bi1:
        sex = st.selectbox(
            "性別",
            ["male", "female"],
            format_func=lambda x: "男性" if x == "male" else "女性",
        )
    with bi2:
        age = st.number_input("年齢（歳）", 20, 95, 60, step=1)
    with bi3:
        bmi_now = st.number_input(
            "BMI", min_value=20.0, max_value=40.0, step=0.5, key="bmi_now_ui"
        )
    st.caption("BMIは記録用です（現行モデルの計算には用いていません）。")

with st.container(border=True):
    st.markdown("#### 血圧・脂質・血糖")
    st.caption("健診・採血結果を見ながら、現在値と目標値を選んでください。")

    st.markdown("**収縮期血圧 (mmHg)**")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.number_input("現在", min_value=90, max_value=240, step=10, key="sbp_now")
    with sc2:
        st.number_input("目標", min_value=90, max_value=240, step=10, key="sbp_tgt")
    sbtn1, sbtn2, _ = st.columns([1, 1, 2])
    sbtn1.button("目標 −10", key="sbp_m10", on_click=_bump, args=("sbp_tgt", -10, 90, 240), use_container_width=True)
    sbtn2.button("目標 ＋10", key="sbp_p10", on_click=_bump, args=("sbp_tgt", 10, 90, 240), use_container_width=True)
    sbp_now = st.session_state["sbp_now"]
    sbp_tgt = st.session_state["sbp_tgt"]

    st.markdown("**LDLコレステロール (mg/dL)**")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.number_input("現在", min_value=40, max_value=300, step=10, key="ldl_now")
    with lc2:
        st.number_input("目標", min_value=40, max_value=300, step=10, key="ldl_tgt")
    lbtn1, lbtn2, _ = st.columns([1, 1, 2])
    lbtn1.button("目標 −30", key="ldl_m30", on_click=_bump, args=("ldl_tgt", -30, 40, 300), use_container_width=True)
    lbtn2.button("目標 −50", key="ldl_m50", on_click=_bump, args=("ldl_tgt", -50, 40, 300), use_container_width=True)
    ldl_now = st.session_state["ldl_now"]
    ldl_tgt = st.session_state["ldl_tgt"]

    st.markdown("**HbA1c (%)**")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.number_input("現在", min_value=5.0, max_value=12.0, step=0.5, key="a1c_now")
    with ac2:
        st.number_input("目標", min_value=5.0, max_value=12.0, step=0.5, key="a1c_tgt")
    abtn1, abtn2, _ = st.columns([1, 1, 2])
    abtn1.button("目標 −0.5", key="a1c_m05", on_click=_bump, args=("a1c_tgt", -0.5, 5.0, 12.0), use_container_width=True)
    abtn2.button("目標 −1.0", key="a1c_m10", on_click=_bump, args=("a1c_tgt", -1.0, 5.0, 12.0), use_container_width=True)
    a1c_now = st.session_state["a1c_now"]
    a1c_tgt = st.session_state["a1c_tgt"]

with st.container(border=True):
    st.markdown("#### 喫煙")
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
        sm1, sm2 = st.columns(2)
        with sm1:
            cigs_per_day = st.number_input("1日あたりの喫煙本数", min_value=0, max_value=40, step=1, key="cigs_per_day")
        with sm2:
            years_smoked = st.number_input("喫煙年数", min_value=0, max_value=60, step=1, key="years_smoked")
        years_since_quit = 0.0
        quit_today = st.checkbox("今日禁煙したと仮定（目標シナリオ）")
    else:
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            cigs_per_day = st.number_input("1日あたりの喫煙本数", min_value=0, max_value=40, step=1, key="cigs_per_day")
        with sm2:
            years_smoked = st.number_input("喫煙年数", min_value=0, max_value=60, step=1, key="years_smoked")
        with sm3:
            years_since_quit = st.number_input("禁煙からの年数（元喫煙者の場合）", min_value=0, max_value=40, step=1, key="years_since_quit")
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
for outcome in ["mortality", "mi", "stroke"]:
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
for outcome in ["mortality", "mi", "stroke"]:
    r = r_by_outcome[outcome]
    diff = r["baseline"] - r["target"]
    st.markdown(
        f"**{labels[outcome]}**（{h}年）: "
        f"現在 **{100 * r['baseline']:.1f}%** → 目標 **{100 * r['target']:.1f}%** · "
        f"差 **{100 * diff:+.1f}%**"
    )
    if outcome == "mortality":
        st.caption("全死亡は、心血管疾患に限らず、がんや他の病気を含むすべての死亡を対象としています。")

st.divider()
st.markdown("### 詳細表示")

detail_blocks = [
    ("mortality", "💀 全死亡", figure_mortality),
    ("mi", "🫀 心筋梗塞", figure_mi),
    ("stroke", "🧠 脳卒中", figure_stroke),
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
    if outcome_key == "mortality":
        st.caption("全死亡は、心血管疾患に限らず、がんや他の病気を含むすべての死亡を対象としています。")
    st.markdown("---")

with st.expander("簡易注記"):
    st.markdown(
        """
- 教育・共有意思決定向けの簡易表示です。医療機器ではありません。
- 本画面は BMI・CKD を含みません（`app_streamlit_outcomes.py` の PC 版で入力できます）。
"""
    )
