import streamlit as st
import plotly.graph_objects as go
import numpy as np
from calc_engine_outcomes import OutcomesEngine

st.set_page_config(
    page_title="JP Outcomes Prevention Simulator (MVP)", 
    layout="wide",
    page_icon="🫀"
)

st.title("🫀📈 アウトカムベース予防シミュレーター（日本、MVP）")
st.caption("教育・共有意思決定のため。医療機器ではありません。")

engine = OutcomesEngine("config.yaml")

with st.sidebar:
    st.subheader("患者プロフィール")
    sex = st.selectbox("性別", ["male","female"], format_func=lambda x: "男性" if x == "male" else "女性")
    age = st.number_input("年齢（歳）", 20, 95, 60, step=1)

    st.subheader("リスク因子（現在 → 目標）")
    sbp_now = st.slider("収縮期血圧 現在 (mmHg)", 90, 200, 150)
    sbp_tgt = st.slider("収縮期血圧 目標 (mmHg)", 90, 160, 130)

    ldl_now = st.slider("LDLコレステロール 現在 (mg/dL)", 50, 250, 160)
    ldl_tgt = st.slider("LDLコレステロール 目標 (mg/dL)", 50, 160, 100)

    a1c_now = st.slider("HbA1c 現在 (%)", 5.0, 12.0, 8.0, step=0.1)
    a1c_tgt = st.slider("HbA1c 目標 (%)", 5.0, 9.0, 7.0, step=0.1)

    st.subheader("喫煙状況")
    smoking_status = st.selectbox("状況", ["never","current","former"], 
                                 format_func=lambda x: {"never": "非喫煙者", "current": "現在喫煙者", "former": "元喫煙者"}[x])
    cigs_per_day = st.slider("1日あたりの喫煙本数", 0, 40, 20)
    years_smoked = st.slider("喫煙年数", 0, 60, 20)
    years_since_quit = st.slider("禁煙からの年数（元喫煙者の場合）", 0, 40, 5)
    quit_today = st.checkbox("今日禁煙したと仮定（目標シナリオ）")

    st.subheader("BMI（任意）")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bmi_now = st.number_input("現在のBMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    with col_b2:
        bmi_target = st.number_input("目標BMI（任意）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)

    st.subheader("CKD（任意）")
    egfr_now = st.number_input("eGFR 現在 (mL/min/1.73m²)", min_value=5.0, max_value=120.0, value=80.0, step=1.0)
    egfr_target = st.number_input("eGFR 目標（任意）", min_value=5.0, max_value=120.0, value=80.0, step=1.0)
    acr_now = st.selectbox("尿アルブミン/蛋白（現在）", ["A1","A2","A3"], index=0,
                           help="A1: 正常/陰性, A2: 微量, A3: 顕性")
    acr_target = st.selectbox("尿アルブミン/蛋白（目標・任意）", ["A1","A2","A3"], index=0)

    st.subheader("予測期間")
    which = st.radio("期間を選択", ["5-year","10-year","20-year","30-year","50-year","Both"], index=2,
                     format_func=lambda x: {"5-year": "5年", "10-year": "10年", "20-year": "20年", "30-year": "30年", "50-year": "50年", "Both": "両方"}[x])

def pct(x): return f"{100*x:.1f}%"

# 累積リスク曲線用のデータを計算
def calculate_cumulative_risk_curves():
    # 選択された期間に応じて年数を設定
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
    else:  # "Both"の場合
        years = 10  # ハザード曲線は10年で表示
    
    # 年単位で計算（ギザギザ防止のため 0.005年刻みはやめる）
    calc_years = np.arange(1, years + 1, 1)  # 1,2,...,years
    
    cumulative_data = {}
    
    for outcome in ['mortality', 'mi', 'stroke']:
        cumulative_data[outcome] = {
            'baseline_cumulative': [],
            'target_cumulative': [],
            'baseline_ci_lower': [],
            'baseline_ci_upper': [],
            'target_ci_lower': [],
            'target_ci_upper': []
        }
        
        # 0年点（原点）を明示してから、年単位の値を積む
        cumulative_data[outcome]['time'] = [0.0]
        cumulative_data[outcome]['baseline_cumulative'] = [0.0]
        cumulative_data[outcome]['target_cumulative'] = [0.0]
        cumulative_data[outcome]['baseline_ci_lower'] = [0.0]
        cumulative_data[outcome]['baseline_ci_upper'] = [0.0]
        cumulative_data[outcome]['target_ci_lower'] = [0.0]
        cumulative_data[outcome]['target_ci_upper'] = [0.0]

        AGE_CAP = 110  # 表示上限
        for y in calc_years:
            age_at_t = age + y
            if age_at_t > AGE_CAP:
                break  # 線はここで切る（NaNではなく時点自体を増やさない）

            res = engine.cumulative_incidence_with_ci(
                outcome, sex, age, int(y),
                sbp_now, sbp_tgt, ldl_now, ldl_tgt,
                a1c_now, a1c_tgt, smoking_status,
                cigs_per_day, years_smoked, years_since_quit,
                    quit_today,
                    bmi_now=bmi_now,
                    bmi_target=bmi_target if bmi_target != bmi_now else None,
                    egfr_now=egfr_now,
                    egfr_target=egfr_target if egfr_target != egfr_now else None,
                    acr_now=acr_now,
                    acr_target=acr_target if acr_target != acr_now else None
            )
            cumulative_data[outcome]['time'].append(float(y))
            cumulative_data[outcome]['baseline_cumulative'].append(res['point']['baseline'] * 100.0)
            cumulative_data[outcome]['target_cumulative'].append(res['point']['target'] * 100.0)
            cumulative_data[outcome]['baseline_ci_lower'].append(res['lower']['baseline'] * 100.0)
            cumulative_data[outcome]['baseline_ci_upper'].append(res['upper']['baseline'] * 100.0)
            cumulative_data[outcome]['target_ci_lower'].append(res['lower']['target'] * 100.0)
            cumulative_data[outcome]['target_ci_upper'].append(res['upper']['target'] * 100.0)
    
    # スプライン補間で曲線を滑らかに（年単位→表示用に高密度化）
    from scipy.interpolate import make_interp_spline
    for outcome in ['mortality', 'mi', 'stroke']:
        ts   = np.array(cumulative_data[outcome]['time'], dtype=float)
        base = np.array(cumulative_data[outcome]['baseline_cumulative'], dtype=float)
        targ = np.array(cumulative_data[outcome]['target_cumulative'], dtype=float)
        bl_l = np.array(cumulative_data[outcome]['baseline_ci_lower'], dtype=float)
        bl_u = np.array(cumulative_data[outcome]['baseline_ci_upper'], dtype=float)
        tg_l = np.array(cumulative_data[outcome]['target_ci_lower'], dtype=float)
        tg_u = np.array(cumulative_data[outcome]['target_ci_upper'], dtype=float)

        if len(ts) >= 4:
            dense_times = np.linspace(ts[0], ts[-1], max(101, int((ts[-1]-ts[0]) * 20)))
            # 本線は3次スプラインで滑らかに、CIは帯の交差を避けるため線形
            base_s = make_interp_spline(ts, base, k=3)(dense_times)
            targ_s = make_interp_spline(ts, targ, k=3)(dense_times)
            bl_l_s = np.interp(dense_times, ts, bl_l)
            bl_u_s = np.interp(dense_times, ts, bl_u)
            tg_l_s = np.interp(dense_times, ts, tg_l)
            tg_u_s = np.interp(dense_times, ts, tg_u)

            cumulative_data[outcome]['time'] = dense_times
            cumulative_data[outcome]['baseline_cumulative'] = base_s
            cumulative_data[outcome]['target_cumulative']   = targ_s
            cumulative_data[outcome]['baseline_ci_lower']   = bl_l_s
            cumulative_data[outcome]['baseline_ci_upper']   = bl_u_s
            cumulative_data[outcome]['target_ci_lower']     = tg_l_s
            cumulative_data[outcome]['target_ci_upper']     = tg_u_s
    
    return cumulative_data

# セッション状態の初期化（先に実施）
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
    st.session_state.cumulative_data = None

# 計算ボタン
if st.button("🔄 リスク計算を実行", type="primary"):
    cumulative_data = calculate_cumulative_risk_curves()
    st.session_state.cumulative_data = cumulative_data
    st.session_state.calculated = True

# 計算結果の表示
if st.session_state.calculated and st.session_state.cumulative_data is not None:
    cumulative_data = st.session_state.cumulative_data
else:
    st.info("👆 上記のパラメータを設定して「リスク計算を実行」ボタンを押してください")
    st.stop()
if which == "Both":
    horizons = [5, 10]
elif which == "5-year":
    horizons = [5]
elif which == "10-year":
    horizons = [10]
elif which == "20-year":
    horizons = [20]
elif which == "30-year":
    horizons = [30]
elif which == "50-year":
    horizons = [50]

# メイン結果表示
st.markdown("### 📊 リスク比較サマリー")
cols = st.columns(3)
labels = {'mi':"心筋梗塞", 'stroke':"脳卒中", 'mortality':"全死亡"}

# 簡潔なサマリー表示
for i, outcome in enumerate(['mortality','mi','stroke']):
    with cols[i]:
        st.subheader(labels[outcome])
        
        # 選択された期間のリスクを計算して表示
        for horizon in horizons:
            r = engine.cumulative_incidence(outcome, sex, age, horizon,
                    sbp_now, sbp_tgt, ldl_now, ldl_tgt, a1c_now, a1c_tgt,
                    smoking_status, cigs_per_day, years_smoked, years_since_quit,
                    assume_quit_today_in_target=quit_today)
            
            # リスク減少の効果を強調
            risk_reduction = r['baseline'] - r['target']
            st.metric(f"{horizon}年リスク減少", f"{100*risk_reduction:.1f}%", 
                     delta=f"現在: {100*r['baseline']:.1f}% → 目標: {100*r['target']:.1f}%")
        if outcome == "mortality":
            st.caption("全死亡は、心血管疾患に限らず、がんや他の病気を含むすべての死亡を対象としています。")

st.divider()

# 累積リスク曲線セクション
st.markdown("### 📈 累積リスク曲線")

# 1. 全死亡の累積リスク曲線（信頼区間付き）
st.markdown("#### 💀 全死亡の累積リスク曲線（95%信頼区間付き）")
fig_mortality_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_mo_t = np.array(cumulative_data['mortality']['time'], dtype=float)
_mo_b = np.array(cumulative_data['mortality']['baseline_cumulative'], dtype=float)
_mo_tg = np.array(cumulative_data['mortality']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_mo_t, cutoff_year, side='right'))

fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[:cut_idx], y=_mo_b[:cut_idx], mode='lines', name='現在のリスク因子',
    line=dict(color='#ef5350', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[cut_idx:], y=_mo_b[cut_idx:], mode='lines', name='現在のリスク因子（≥85歳推定域）',
    line=dict(color='rgba(239,83,80,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[:cut_idx], y=_mo_tg[:cut_idx], mode='lines', name='目標達成時',
    line=dict(color='#26a69a', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[cut_idx:], y=_mo_tg[cut_idx:], mode='lines', name='目標達成時（≥85歳推定域）',
    line=dict(color='rgba(38,166,154,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))

# 現在のリスク因子の信頼区間帯
fig_mortality_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mortality']['time'],
    y=cumulative_data['mortality']['baseline_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_mortality_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mortality']['time'],
    y=cumulative_data['mortality']['baseline_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='現在のリスク因子 95%CI',
    fillcolor='rgba(239,83,80,0.2)'
))

# 目標達成時の信頼区間帯
fig_mortality_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mortality']['time'],
    y=cumulative_data['mortality']['target_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_mortality_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mortality']['time'],
    y=cumulative_data['mortality']['target_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='目標達成時 95%CI',
    fillcolor='rgba(38,166,154,0.2)'
))

fig_mortality_cumulative.update_layout(
    title="全死亡の累積リスク曲線（95%信頼区間付き）",
    xaxis_title="年数",
    yaxis_title="累積リスク（%）",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

# 線を滑らかにする設定
for trace in fig_mortality_cumulative.data:
    if trace.mode == 'lines' and hasattr(trace, 'name') and trace.name and '95%CI' not in trace.name:
        # メインの線のみを滑らかに（信頼区間帯は除外）
        trace.update(line=dict(smoothing=1.0, shape='spline'))

st.plotly_chart(fig_mortality_cumulative, use_container_width=True)
st.caption("全死亡は、心血管疾患に限らず、がんや他の病気を含むすべての死亡を対象としています。")


# 2. 心筋梗塞の累積リスク曲線（信頼区間付き）
st.markdown("#### 🫀 心筋梗塞の累積リスク曲線（95%信頼区間付き）")
fig_mi_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_mi_t = np.array(cumulative_data['mi']['time'], dtype=float)
_mi_b = np.array(cumulative_data['mi']['baseline_cumulative'], dtype=float)
_mi_tg = np.array(cumulative_data['mi']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_mi_t, cutoff_year, side='right'))

# baseline: ～85歳
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[:cut_idx], y=_mi_b[:cut_idx], mode='lines', name='現在のリスク因子',
    line=dict(color='#ff6b6b', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
# baseline: 85歳～（薄色）
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[cut_idx:], y=_mi_b[cut_idx:], mode='lines', name='現在のリスク因子（≥85歳推定域）',
    line=dict(color='rgba(255,107,107,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
# target: ～85歳
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[:cut_idx], y=_mi_tg[:cut_idx], mode='lines', name='目標達成時',
    line=dict(color='#4ecdc4', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
# target: 85歳～（薄色）
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[cut_idx:], y=_mi_tg[cut_idx:], mode='lines', name='目標達成時（≥85歳推定域）',
    line=dict(color='rgba(78,205,196,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))

# 現在のリスク因子の信頼区間帯
fig_mi_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mi']['time'],
    y=cumulative_data['mi']['baseline_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_mi_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mi']['time'],
    y=cumulative_data['mi']['baseline_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='現在のリスク因子 95%CI',
    fillcolor='rgba(255,107,107,0.2)'
))

# 目標達成時の信頼区間帯
fig_mi_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mi']['time'],
    y=cumulative_data['mi']['target_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_mi_cumulative.add_trace(go.Scatter(
    x=cumulative_data['mi']['time'],
    y=cumulative_data['mi']['target_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='目標達成時 95%CI',
    fillcolor='rgba(78,205,196,0.2)'
))

fig_mi_cumulative.update_layout(
    title="心筋梗塞の累積リスク曲線（95%信頼区間付き）",
    xaxis_title="年数",
    yaxis_title="累積リスク（%）",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

# 線を滑らかにする設定
for trace in fig_mi_cumulative.data:
    if trace.mode == 'lines' and hasattr(trace, 'name') and trace.name and '95%CI' not in trace.name:
        # メインの線のみを滑らかに（信頼区間帯は除外）
        trace.update(line=dict(smoothing=1.0, shape='spline'))

st.plotly_chart(fig_mi_cumulative, use_container_width=True)

# 3. 脳卒中の累積リスク曲線（信頼区間付き）
st.markdown("#### 🧠 脳卒中の累積リスク曲線（95%信頼区間付き）")
fig_stroke_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_st_t = np.array(cumulative_data['stroke']['time'], dtype=float)
_st_b = np.array(cumulative_data['stroke']['baseline_cumulative'], dtype=float)
_st_tg = np.array(cumulative_data['stroke']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_st_t, cutoff_year, side='right'))

fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[:cut_idx], y=_st_b[:cut_idx], mode='lines', name='現在のリスク因子',
    line=dict(color='#ffa726', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[cut_idx:], y=_st_b[cut_idx:], mode='lines', name='現在のリスク因子（≥85歳推定域）',
    line=dict(color='rgba(255,167,38,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[:cut_idx], y=_st_tg[:cut_idx], mode='lines', name='目標達成時',
    line=dict(color='#66bb6a', width=3), showlegend=True,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[cut_idx:], y=_st_tg[cut_idx:], mode='lines', name='目標達成時（≥85歳推定域）',
    line=dict(color='rgba(102,187,106,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f}年: %{y:.2f}%<extra></extra>'
))

# 現在のリスク因子の信頼区間帯
fig_stroke_cumulative.add_trace(go.Scatter(
    x=cumulative_data['stroke']['time'],
    y=cumulative_data['stroke']['baseline_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_stroke_cumulative.add_trace(go.Scatter(
    x=cumulative_data['stroke']['time'],
    y=cumulative_data['stroke']['baseline_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='現在のリスク因子 95%CI',
    fillcolor='rgba(255,167,38,0.2)'
))

# 目標達成時の信頼区間帯
fig_stroke_cumulative.add_trace(go.Scatter(
    x=cumulative_data['stroke']['time'],
    y=cumulative_data['stroke']['target_ci_upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

fig_stroke_cumulative.add_trace(go.Scatter(
    x=cumulative_data['stroke']['time'],
    y=cumulative_data['stroke']['target_ci_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='目標達成時 95%CI',
    fillcolor='rgba(102,187,106,0.2)'
))

fig_stroke_cumulative.update_layout(
    title="脳卒中の累積リスク曲線（95%信頼区間付き）",
    xaxis_title="年数",
    yaxis_title="累積リスク（%）",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

# 線を滑らかにする設定
for trace in fig_stroke_cumulative.data:
    if trace.mode == 'lines' and hasattr(trace, 'name') and trace.name and '95%CI' not in trace.name:
        # メインの線のみを滑らかに（信頼区間帯は除外）
        trace.update(line=dict(smoothing=1.0, shape='spline'))

st.plotly_chart(fig_stroke_cumulative, use_container_width=True)

st.divider()
with st.expander("出典・注記"):
    st.markdown("""
**ベースライン（日本）:**
- 心筋梗塞: 宮城AMIレジストリー（2014年近似値、CSVに含む。正確な値が利用可能になったら置き換えてください）
- 脳卒中: 滋賀脳卒中レジストリー（CSVに代表的な値）
- 死亡率: 日本R5生命表 `qx` を **CSVの `qx` 列として直接利用**（推奨）。CSVが無い場合は一時的にGompertz近似を使用します。
    """)

# 一次予防モデル脚注（整形済み）
with st.expander("一次予防モデル脚注（心筋梗塞・脳卒中・全死亡）"):
    st.markdown("""
**目的**: 外来で取得できる因子（SBP・LDL-C・HbA1c・喫煙・BMI・CKD[eGFR/アルブミン尿]）の是正による主要アウトカム（MI/Stroke/All-cause mortality）の累積リスク差を可視化。

**ベースライン発症率**: 年齢・性別別CSVを補間して使用（死亡は生命表`qx`、MI/Strokeは年率を確率化）。

**累積計算（離散時間）**:
- 年ごとに age=t年後の年齢へ更新
- その年の年次発症確率: q_t = baseline(age_t, sex, outcome) × RR_total(age_t)
- 累積: CumRisk_{t+1} = CumRisk_t + (1 − CumRisk_t) × q_t

**年齢減衰の考え方**: 相対効果は高齢で小さくなる傾向があるため、各因子の ln(RR) に係数 α(age) を掛けて調整。≥85歳は推定域として保守的に扱う（相対効果は弱め/ゼロ近傍）。

1) SBP（収縮期血圧）
- 単位効果: 5 mmHg低下ごと HR≈0.91（Stroke寄りに強め、MIはやや弱め）
- 年齢減衰: α_SBP(age)=1.0（≤75）→線形→0.0（85）→以降0.0

2) LDL-C
- 単位効果: 1 mmol/L（≈38.7 mg/dL）低下ごと HR≈0.77
- 年齢減衰: α_LDL(age)=1.0（≤85）→0.7（90）→以降0.7（軽い減衰）

3) HbA1c（宏血管想定）
- 方向性: 1%低下でRR<1だが控えめ。年齢減衰: α_A1c(age)=1.0（≤75）→0.0（85）→以降0.0
- 微小血管は本モデル外（将来拡張）。

4) 喫煙
- 現喫煙でHR上昇。禁煙後は HR(y)=1+(HR0−1)×exp(−k·y)（k≈0.15–0.2）で低下。
- 年齢減衰は弱め（相対差は広い年齢で持続）。

5) BMI（U字＋年齢シフト）
- 最適BMI（谷）を年齢で 23.5→26.5 にシフト（40→80歳）。
- 高BMI側: 若年ほど強く、超高齢で中立化。低BMI側: 高齢ほど不利。
- 1BMIあたりの連続モデル: β=ln(RR5)/5, RR=exp(β×ΔBMI)。年次に再評価して乗算。極端入力は 0.5–2.0 にクリップ。

6) CKD（eGFR/アルブミン尿）
- 点推定RR: eGFR ≥60=1.0、45–59=1.30、<45=1.80。ACR: A1=1.0、A2=1.35、A3=1.90。
- 結合は初期は max(rr_eGFR, rr_ACR) を採用（上級設定で乗算＋上限可）。
- 年齢減衰（lnRR×α）: A2/A3: 1.0→0.85（85）→0.80（>85）/ 低eGFR単独: 1.0→0.80→0.70 / 両者あり: 1.0→0.90→0.85。
- アウトカム別調整: MI×0.8 / Stroke×1.0 / Mortality×1.1。

**出力時の注記**
- エビデンス範囲: 相対効果は概ね〜85歳までが実証域。85歳超は推定域として保守的に補正（本グラフでは≥85歳の線色を薄色表示）。
- 二重カウント回避: 減塩などはSBP経由で反映。BMIと腹囲は同時に強くは使わない。
- 絶対 vs 相対: 高齢では相対差は縮むが、絶対差（ARR）は維持・増大し得る。

**代表参照（例）**
- 血圧: SPRINT（NEJM 2015）、HYVET（NEJM 2008）、BPLTTC / Rahimi et al.（Lancet 2021）
- 脂質: CTT 共同解析（Lancet 一連）、WOSCOPS（NEJM 1995）、ASCOT-LLA（Lancet 2003）、JUPITER（NEJM 2008）
- 血糖: UKPDS、ADVANCE（NEJM 2008）、ACCORD（NEJM 2008/2010）、Selvinほか（Diabetes Care）
- 喫煙: INTERHEART（Lancet 2004）、各国コホート/公衆衛生総報
- BMI: Prospective Studies Collaboration（Lancet 2009）
- CKD: CKD-PC（Lancet 2010/2012 ほか）、HOPE/RENAAL/IDNT（NEJM 2000–2001）、SPRINT/HYVET
    """)
