import streamlit as st
import plotly.graph_objects as go
import numpy as np
from calc_engine_outcomes import OutcomesEngine

st.set_page_config(
    page_title="選べる未来（一次予防・共有用）",
    layout="centered",
    page_icon="🫀",
)

st.markdown(
    """
    <style>
    .masthead {
        text-align: center;
        padding: 2.4rem 1.2rem 2.1rem 1.2rem;
        margin-bottom: 1.75rem;
        background: linear-gradient(165deg, #faf7f2 0%, #f3efe8 48%, #ebe4db 100%);
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(45, 40, 36, 0.08);
        border: 1px solid rgba(55, 48, 42, 0.06);
    }
    .masthead h1 {
        font-size: clamp(1.55rem, 4.5vw, 2.05rem);
        font-weight: 750;
        letter-spacing: 0.04em;
        color: #1c1917;
        margin: 0 0 0.85rem 0;
        line-height: 1.28;
    }
    .masthead p {
        font-size: clamp(0.95rem, 3.2vw, 1.05rem);
        color: #3f3a36;
        line-height: 1.75;
        margin: 0;
        max-width: 26rem;
        margin-left: auto;
        margin-right: auto;
    }
    .future-card {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0 0 0.15rem 0;
        margin-bottom: 0;
        box-shadow: none;
        min-height: auto;
    }
    .fc-emoji {
        font-size: 1.65rem;
        line-height: 1;
        margin-bottom: 0.35rem;
    }
    .fc-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1c1917;
        margin: 0 0 0.25rem 0;
        line-height: 1.35;
    }
    .fc-sub {
        font-size: 0.82rem;
        color: #5c534c;
        line-height: 1.45;
        margin: 0 0 0.5rem 0;
    }
    .pill {
        display: inline-block;
        padding: 0.38rem 0.85rem;
        margin: 0.2rem 0.28rem 0.15rem 0;
        background: #f4efe8;
        border-radius: 999px;
        font-size: 0.9rem;
        color: #2d2824;
        border: 1px solid rgba(55, 48, 42, 0.08);
    }
    .outcome-head {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        margin-bottom: 0.35rem;
    }
    .outcome-ico {
        font-size: 1.35rem;
        line-height: 1.2;
    }
    .outcome-name {
        font-size: 1.08rem;
        font-weight: 700;
        color: #1c1917;
        margin: 0;
        line-height: 1.3;
    }
    .outcome-sub {
        font-size: 0.84rem;
        color: #5c534c;
        margin: 0.15rem 0 0.6rem 1.9rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="masthead">
  <h1>自分の未来を、自分で選ぶ。</h1>
  <p>血圧やコレステロールを下げることが、ゴールではありません。<br>
  その先で、何を続けたいか。<br>
  まずはそこから、考えてみましょう。</p>
</div>
    """,
    unsafe_allow_html=True,
)

engine = OutcomesEngine("config.yaml")

if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.cumulative_data = None

FUTURE_GOALS = [
    {
        "key": "travel",
        "emoji": "✈️",
        "title": "旅行に行きたい",
        "sub": "もう一度、行きたい場所へ。",
    },
    {
        "key": "family",
        "emoji": "👨‍👩‍👧‍👦",
        "title": "家族や孫と過ごしたい",
        "sub": "大切な人との時間を、これからも。",
    },
    {
        "key": "work",
        "emoji": "💼",
        "title": "仕事を続けたい",
        "sub": "自分らしく働き続けるために。",
    },
    {
        "key": "hobby",
        "emoji": "🎾",
        "title": "趣味やスポーツを続けたい",
        "sub": "好きなことを、あきらめないために。",
    },
    {
        "key": "walk",
        "emoji": "🚶",
        "title": "自分の足で歩き続けたい",
        "sub": "行きたい場所へ、自分の足で。",
    },
    {
        "key": "home",
        "emoji": "🏠",
        "title": "入院せずに暮らしたい",
        "sub": "できるだけ、いつもの生活を続けるために。",
    },
    {
        "key": "vitality",
        "emoji": "🌱",
        "title": "長く元気でいたい",
        "sub": "年齢を重ねても、自分らしく。",
    },
    {
        "key": "food",
        "emoji": "🍽️",
        "title": "好きな食事を楽しみたい",
        "sub": "楽しみを残しながら、健康も考える。",
    },
]


def _goal_line(g):
    return f"{g['emoji']} {g['title']}"


def _selected_future_keys():
    return [g["key"] for g in FUTURE_GOALS if st.session_state.get(f"fg_{g['key']}", False)]


def _render_chosen_future_pills(sel_keys):
    if not sel_keys:
        return
    parts = []
    for g in FUTURE_GOALS:
        if g["key"] in sel_keys:
            parts.append(f'<span class="pill">{_goal_line(g)}</span>')
    st.markdown("".join(parts), unsafe_allow_html=True)


BRIDGE_TO_INPUTS = {
    "travel": "旅行や外出を続けるために、いまの状態を見てみましょう。",
    "family": "大切な人との時間を長く楽しむために、いまの状態を見てみましょう。",
    "work": "自分らしく働き続けるために、いまの状態を見てみましょう。",
    "hobby": "好きなことを続けるために、いまの状態を見てみましょう。",
    "walk": "行きたい場所へ自分の足で向かうために、いまの状態を見てみましょう。",
    "home": "いつもの生活を続けるために、いまの状態を見てみましょう。",
    "vitality": "これからも自分らしく過ごすために、いまの状態を見てみましょう。",
    "food": "楽しみを残しながら健康も守るために、いまの状態を見てみましょう。",
}


def _bridge_to_inputs_text(sel_keys):
    """選択の先頭（フォーム上の並び）を代表に橋渡し文を返す。UIのみ。"""
    if not sel_keys:
        return "気になるものがあれば、あとから選べます。まずはいまの状態を見てみましょう。"
    return BRIDGE_TO_INPUTS.get(sel_keys[0], "その未来を支えるために、いまの状態を見てみましょう。")


MI_FUTURE_TIE = {
    "family": "大切な人との時間を、突然途切れさせないために。",
    "travel": "行きたい場所へ行ける体力を保つために。",
    "work": "突然の休職や通院増加を減らすために。",
    "hobby": "好きなことを続けられる体力を守るために。",
    "walk": "外出できる体力を保つために。",
    "home": "突然の入院を減らすために。",
    "vitality": "自分らしく過ごせる時間を守るために。",
    "food": "楽しみを残しながら暮らすために。",
    "default": "突然の入院や体力低下に備える見通しです。",
}

STROKE_FUTURE_TIE = {
    "family": "家族との日常を、自分らしく続けるために。",
    "travel": "遠出や外出の自由を保つために。",
    "work": "働く力や生活リズムを守るために。",
    "hobby": "好きなことを続ける体の自由を守るために。",
    "walk": "歩く力を守るために。",
    "home": "入院やリハビリを減らすために。",
    "vitality": "生活の自立を保つために。",
    "food": "食べる楽しみを守るために。",
    "default": "歩く・話す・食べる力への影響に備える見通しです。",
}

MORTALITY_FUTURE_TIE = {
    "family": "大切な人との時間を長く楽しむために。",
    "travel": "これからも行きたい場所へ向かうために。",
    "work": "自分らしく働く時間を保つために。",
    "hobby": "好きなことを続ける時間を保つために。",
    "walk": "自分の足で行きたい場所へ向かうために。",
    "home": "いつもの生活を続けるために。",
    "vitality": "これからも自分らしく過ごすために。",
    "food": "楽しみを残しながら暮らすために。",
    "default": "これからの時間の見通しを考えるための目安です。",
}


def _future_tie_lines(sel_keys, tie_map, max_lines=2):
    """選択に応じた接続文を最大 max_lines 件。UIのみ。"""
    if not sel_keys:
        return [tie_map["default"]]
    lines = []
    for g in FUTURE_GOALS:
        k = g["key"]
        if k in sel_keys and k in tie_map:
            t = tie_map[k]
            if t not in lines:
                lines.append(t)
            if len(lines) >= max_lines:
                return lines
    return lines if lines else [tie_map["default"]]


OUTCOME_NEXT_TALK = {
    "mi": "突然の入院や体力低下を減らすために、血圧やLDLの目標を一緒に決める。",
    "stroke": "歩く力や生活の自由を守るために、血圧や血糖、禁煙について相談する。",
    "mortality": "自分らしい時間を保つために、無理なく続けられる治療や生活習慣を相談する。",
}


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


def figure_mi(cumulative_data, age, *, patient_legend=False, patient_title=False):
    fig = go.Figure()
    _t = np.array(cumulative_data["mi"]["time"], dtype=float)
    _b = np.array(cumulative_data["mi"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["mi"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    lab_b = ("このままの場合", "このままの場合（85歳以降は推定）")
    lab_t = ("目標に近づいた場合", "目標に近づいた場合（85歳以降は推定）")
    if not patient_legend:
        lab_b = ("現在の状態", "現在の状態（85歳以降は推定）")
        lab_t = ("目標の状態", "目標の状態（85歳以降は推定）")

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name=lab_b[0],
            legendgroup="base",
            line=dict(color="#ff6b6b", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name=lab_b[1],
            legendgroup="base",
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
            name=lab_t[0],
            legendgroup="tgt",
            line=dict(color="#4ecdc4", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name=lab_t[1],
            legendgroup="tgt",
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
            name="このまま 95%CI",
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
            name="目標側 95%CI",
            fillcolor="rgba(78,205,196,0.2)",
            showlegend=False,
        )
    )
    _title = "心筋梗塞 ― 時間とともにみた見通し（累積%）"
    _layout = dict(
        xaxis_title="年数",
        yaxis_title="累積の見通し（%）",
        height=360 if patient_legend else 320,
        showlegend=patient_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50 if patient_legend else 44, b=60),
    )
    if patient_title:
        fig.update_layout(**_layout)
    else:
        fig.update_layout(title=dict(text=_title), **_layout)
    _smooth_main_lines(fig)
    return fig


def figure_stroke(cumulative_data, age, *, patient_legend=False, patient_title=False):
    fig = go.Figure()
    _t = np.array(cumulative_data["stroke"]["time"], dtype=float)
    _b = np.array(cumulative_data["stroke"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["stroke"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    lab_b = ("このままの場合", "このままの場合（85歳以降は推定）")
    lab_t = ("目標に近づいた場合", "目標に近づいた場合（85歳以降は推定）")
    if not patient_legend:
        lab_b = ("現在の状態", "現在の状態（85歳以降は推定）")
        lab_t = ("目標の状態", "目標の状態（85歳以降は推定）")

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name=lab_b[0],
            legendgroup="base",
            line=dict(color="#ffa726", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name=lab_b[1],
            legendgroup="base",
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
            name=lab_t[0],
            legendgroup="tgt",
            line=dict(color="#66bb6a", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name=lab_t[1],
            legendgroup="tgt",
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
            name="このまま 95%CI",
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
            name="目標側 95%CI",
            fillcolor="rgba(102,187,106,0.2)",
            showlegend=False,
        )
    )
    _title = "脳卒中 ― 時間とともにみた見通し（累積%）"
    _layout = dict(
        xaxis_title="年数",
        yaxis_title="累積の見通し（%）",
        height=360 if patient_legend else 320,
        showlegend=patient_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50 if patient_legend else 44, b=60),
    )
    if patient_title:
        fig.update_layout(**_layout)
    else:
        fig.update_layout(title=dict(text=_title), **_layout)
    _smooth_main_lines(fig)
    return fig


def figure_mortality(cumulative_data, age, *, patient_legend=False, patient_title=False):
    fig = go.Figure()
    _t = np.array(cumulative_data["mortality"]["time"], dtype=float)
    _b = np.array(cumulative_data["mortality"]["baseline_cumulative"], dtype=float)
    _tg = np.array(cumulative_data["mortality"]["target_cumulative"], dtype=float)
    cutoff_year = max(0.0, 85.0 - float(age))
    cut_idx = int(np.searchsorted(_t, cutoff_year, side="right"))

    lab_b = ("このままの場合", "このままの場合（85歳以降は推定）")
    lab_t = ("目標に近づいた場合", "目標に近づいた場合（85歳以降は推定）")
    if not patient_legend:
        lab_b = ("現在の状態", "現在の状態（85歳以降は推定）")
        lab_t = ("目標の状態", "目標の状態（85歳以降は推定）")

    fig.add_trace(
        go.Scatter(
            x=_t[:cut_idx],
            y=_b[:cut_idx],
            mode="lines",
            name=lab_b[0],
            legendgroup="base",
            line=dict(color="#ef5350", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_b[cut_idx:],
            mode="lines",
            name=lab_b[1],
            legendgroup="base",
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
            name=lab_t[0],
            legendgroup="tgt",
            line=dict(color="#26a69a", width=2),
            showlegend=patient_legend,
            hovertemplate="%{x:.1f}年: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_t[cut_idx:],
            y=_tg[cut_idx:],
            mode="lines",
            name=lab_t[1],
            legendgroup="tgt",
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
            name="このまま 95%CI",
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
            name="目標側 95%CI",
            fillcolor="rgba(38,166,154,0.2)",
            showlegend=False,
        )
    )
    _title = "全死亡 ― 時間とともにみた見通し（累積%）"
    _layout = dict(
        xaxis_title="年数",
        yaxis_title="累積の見通し（%）",
        height=360 if patient_legend else 320,
        showlegend=patient_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50 if patient_legend else 44, b=60),
    )
    if patient_title:
        fig.update_layout(**_layout)
    else:
        fig.update_layout(title=dict(text=_title), **_layout)
    _smooth_main_lines(fig)
    return fig


st.markdown("## この先も、続けたいこと")
st.caption("あなたにとって大切な未来を選んでください。あとから変えても大丈夫です。")

gcols = st.columns(2)
for i, g in enumerate(FUTURE_GOALS):
    with gcols[i % 2]:
        with st.container(border=True):
            st.markdown(
                f"""
<div class="future-card">
  <div class="fc-emoji">{g["emoji"]}</div>
  <p class="fc-title">{g["title"]}</p>
  <p class="fc-sub">{g["sub"]}</p>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.checkbox("選ぶ", key=f"fg_{g['key']}")

st.markdown("### あなたが選んだ未来")
sel_keys_preview = _selected_future_keys()
if sel_keys_preview:
    _render_chosen_future_pills(sel_keys_preview)

st.markdown(
    f'<p style="color:#3f3a36;font-size:0.98rem;margin:0.45rem 0 1.1rem 0;">{_bridge_to_inputs_text(sel_keys_preview)}</p>',
    unsafe_allow_html=True,
)

st.markdown("### その未来を支えるために、いまの状態を見てみる")
st.caption("このままだとどうか。少し変えるとどうか。いまの値と目標を並べて比べます。")

with st.container(border=True):
    st.markdown("#### 基本情報")
    sex = st.selectbox(
        "性別",
        ["male", "female"],
        format_func=lambda x: "男性" if x == "male" else "女性",
    )
    age = st.number_input("年齢（歳）", 20, 95, 60, step=1)

with st.container(border=True):
    st.markdown("#### 血圧・脂質・血糖")
    if sel_keys_preview:
        st.caption("大切にしたい未来に向けて、まず血圧・LDL・HbA1cを見てみます。")
    else:
        st.caption("いまの値と、目指したい値を並べて入力します。")
    sbp_now = st.slider("収縮期血圧：いま", 90, 200, 150)
    sbp_tgt = st.slider("収縮期血圧：目標", 90, 160, 130)

    ldl_now = st.slider("LDLコレステロール：いま", 50, 250, 160)
    ldl_tgt = st.slider("LDLコレステロール：目標", 50, 160, 100)

    a1c_now = st.slider("HbA1c：いま", 5.0, 12.0, 8.0, step=0.1)
    a1c_tgt = st.slider("HbA1c：目標", 5.0, 9.0, 7.0, step=0.1)

with st.container(border=True):
    st.markdown("#### 喫煙")
    smoking_status = st.selectbox(
        "喫煙の状態",
        ["never", "current", "former"],
        format_func=lambda x: {"never": "非喫煙", "current": "喫煙中", "former": "過去に喫煙"}[x],
    )
    if smoking_status == "never":
        cigs_per_day = 0
        years_smoked = 0.0
        years_since_quit = 0.0
        quit_today = False
    elif smoking_status == "current":
        cigs_per_day = st.slider("1日あたりの本数", 0, 40, 20)
        years_smoked = st.slider("喫煙年数", 0, 60, 20)
        years_since_quit = 0.0
        quit_today = st.checkbox("「いまから禁煙を始めた」と仮定して、目標側の未来も見る")
    else:
        cigs_per_day = st.slider("1日あたりの本数（喫煙していたころ）", 0, 40, 20)
        years_smoked = st.slider("喫煙年数", 0, 60, 20)
        years_since_quit = st.slider("禁煙してからの年数", 0, 40, 5)
        quit_today = False

with st.container(border=True):
    st.markdown("#### 見通しの期間")
    st.caption("どのくらい先まで見てみますか？")
    which = st.radio(
        "期間",
        ["5-year", "10-year", "20-year", "30-year", "50-year"],
        index=2,
        label_visibility="collapsed",
        format_func=lambda x: {
            "5-year": "5年先まで",
            "10-year": "10年先まで",
            "20-year": "20年先まで",
            "30-year": "30年先まで",
            "50-year": "50年先まで",
        }[x],
    )

if st.button("この先の見通しを見る", type="primary"):
    st.session_state.cumulative_data = calculate_cumulative_curves()
    st.session_state.calculated = True

if not (st.session_state.calculated and st.session_state.cumulative_data is not None):
    st.stop()

cumulative_data = st.session_state.cumulative_data

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

st.divider()
st.markdown("### あなたが大切にしたい未来")
sel_keys = _selected_future_keys()
if sel_keys:
    _render_chosen_future_pills(sel_keys)
else:
    st.caption("まだ選ばなくても大丈夫です。気になるものがあれば、あとから選べます。")

st.markdown("## このままだと。少し変えると。")

OUTCOME_BLOCKS = [
    {
        "key": "mi",
        "ico": "❤️",
        "title": "心筋梗塞",
        "message": "突然、日常が止まることがあります。",
        "explanation": [
            "心筋梗塞は、突然死や緊急入院につながることがあります。",
            "治療後も、心不全、息切れ、体力低下で、これまで通りに動けなくなることがあります。",
        ],
        "tie_map": MI_FUTURE_TIE,
        "fig": figure_mi,
    },
    {
        "key": "stroke",
        "ico": "🧠",
        "title": "脳卒中",
        "message": "できていたことが、急に難しくなることがあります。",
        "explanation": [
            "脳卒中は、命に関わることがあります。",
            "歩く、話す、食べる、働く力に影響し、リハビリや介助が必要になることがあります。",
        ],
        "tie_map": STROKE_FUTURE_TIE,
        "fig": figure_stroke,
    },
    {
        "key": "mortality",
        "ico": "🌱",
        "title": "寿命に関わる見通し",
        "message": "自分らしく過ごせる時間を守るために。",
        "explanation": [
            "これは「あと何年」と決めるものではありません。",
            "生活習慣や治療の選択で、これからの見通しがどう変わるかを考えるための目安です。",
        ],
        "tie_map": MORTALITY_FUTURE_TIE,
        "fig": figure_mortality,
    },
]

for ob in OUTCOME_BLOCKS:
    outcome = ob["key"]
    r = r_by_outcome[outcome]
    diff = r["baseline"] - r["target"]
    pct_b = 100 * r["baseline"]
    pct_t = 100 * r["target"]
    diff_pts = 100 * diff

    st.markdown(
        f"""
<div class="outcome-head">
  <span class="outcome-ico">{ob["ico"]}</span>
  <p class="outcome-name">{ob["title"]}</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**{ob['message']}**")
    for line in ob["explanation"]:
        st.caption(line)

    if sel_keys:
        for tie_line in _future_tie_lines(sel_keys, ob["tie_map"], max_lines=2):
            st.caption(tie_line)

    fig_p = ob["fig"](cumulative_data, age, patient_legend=True, patient_title=True)
    st.plotly_chart(fig_p, use_container_width=True)

    with st.container(border=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("このままの場合", f"{pct_b:.1f}%")
        with s2:
            st.metric("目標に近づいた場合", f"{pct_t:.1f}%")
        with s3:
            if diff > 0.0005:
                st.metric("差", f"{diff_pts:.1f}ポイント低い見通し")
            else:
                st.metric("差", "差は小さめです")
        st.markdown("##### 次に話すこと")
        st.markdown(f"- {OUTCOME_NEXT_TALK[outcome]}")

with st.expander("医療者向け：グラフと詳しい数値", expanded=False):
    st.caption(
        f"※{h}年時点の累積％の目安。100人に似た場合は、およそ同じ数字の人分と読み替えられます。"
    )
    st.caption("時系列の線は、上の共有表示を参照してください。")
    detail_blocks = [
        ("mi", "心筋梗塞"),
        ("stroke", "脳卒中"),
        ("mortality", "全死亡"),
    ]

    for outcome_key, heading in detail_blocks:
        st.subheader(heading)
        r = r_by_outcome[outcome_key]
        diff = r["baseline"] - r["target"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("いま（%）", f"{100 * r['baseline']:.1f}%")
        with c2:
            st.metric("目標（%）", f"{100 * r['target']:.1f}%")
        with c3:
            st.metric("差（ポイント）", f"{100 * diff:.1f}")
        if outcome_key == "mortality":
            st.caption("全死亡は、心血管疾患に限らず、がんや他の病気を含むすべての死亡を対象としています。")
        st.markdown("---")

with st.expander("この画面について", expanded=False):
    st.markdown(
        """
この画面は、生活習慣や治療について医療者と話し合うための参考情報です。  
表示される数値は、入力された条件に基づく推定です。  
診断や治療方針は、担当医と相談して決めてください。
"""
    )
