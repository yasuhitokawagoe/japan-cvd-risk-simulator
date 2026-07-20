import streamlit as st
import plotly.graph_objects as go
import numpy as np
from calc_engine_outcomes import OutcomesEngine

st.set_page_config(
    page_title="JP Outcomes Prevention Simulator (MVP)", 
    layout="wide",
    page_icon="🫀"
)

st.title("🫀📈 Japan Cardiovascular Risk Prediction Simulator")
st.caption("For education and shared decision-making. This is not a medical device. Medication version: https://japan-cvd-risk-simulator-meds-fm.streamlit.app/")

engine = OutcomesEngine("config.yaml")

with st.sidebar:
    st.subheader("Patient Profile")
    sex = st.selectbox("Sex", ["male","female"], format_func=lambda x: "Male" if x == "male" else "Female")
    age = st.number_input("Age (years)", 20, 95, 60, step=1)

    st.subheader("Risk Factors (Current → Target)")
    sbp_now = st.slider("Current systolic blood pressure (mmHg)", 90, 200, 150)
    sbp_tgt = st.slider("Target systolic blood pressure (mmHg)", 90, 160, 130)

    ldl_now = st.slider("Current LDL cholesterol (mg/dL)", 50, 250, 160)
    ldl_tgt = st.slider("Target LDL cholesterol (mg/dL)", 50, 160, 100)

    a1c_now = st.slider("Current HbA1c (%)", 5.0, 12.0, 8.0, step=0.1)
    a1c_tgt = st.slider("Target HbA1c (%)", 5.0, 9.0, 7.0, step=0.1)

    st.subheader("Smoking Status")
    smoking_status = st.selectbox("Status", ["never","current","former"],
                                 format_func=lambda x: {"never": "Never smoked", "current": "Current smoker", "former": "Former smoker"}[x])
    cigs_per_day = st.slider("Cigarettes per day", 0, 40, 20)
    years_smoked = st.slider("Years smoked", 0, 60, 20)
    years_since_quit = st.slider("Years since quitting (former smokers)", 0, 40, 5)
    quit_today = st.checkbox("Assume quitting today (target scenario)")

    st.subheader("BMI (optional)")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bmi_now = st.number_input("Current BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    with col_b2:
        bmi_target = st.number_input("Target BMI (optional)", min_value=10.0, max_value=50.0, value=24.0, step=0.1)

    st.subheader("CKD (optional)")
    egfr_now = st.number_input("Current eGFR (mL/min/1.73m²)", min_value=5.0, max_value=120.0, value=80.0, step=1.0)
    egfr_target = st.number_input("Target eGFR (optional)", min_value=5.0, max_value=120.0, value=80.0, step=1.0)
    acr_now = st.selectbox("Current urine albumin/protein category", ["A1","A2","A3"], index=0,
                           help="A1: normal/negative, A2: moderately increased, A3: severely increased")
    acr_target = st.selectbox("Target urine albumin/protein category (optional)", ["A1","A2","A3"], index=0)

    st.subheader("Prediction Horizon")
    which = st.radio("Select a time horizon", ["5-year","10-year","20-year","30-year","50-year","Both"], index=2,
                     format_func=lambda x: {"5-year": "5 years", "10-year": "10 years", "20-year": "20 years", "30-year": "30 years", "50-year": "50 years", "Both": "5 and 10 years"}[x])

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
if st.button("🔄 Calculate Risk", type="primary"):
    cumulative_data = calculate_cumulative_risk_curves()
    st.session_state.cumulative_data = cumulative_data
    st.session_state.calculated = True

# 計算結果の表示
if st.session_state.calculated and st.session_state.cumulative_data is not None:
    cumulative_data = st.session_state.cumulative_data
else:
    st.info("👆 Set the parameters above, then select Calculate Risk.")
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
st.markdown("### 📊 Risk Comparison Summary")
cols = st.columns(3)
labels = {'mi':"Myocardial Infarction", 'stroke':"Stroke", 'mortality':"All-Cause Mortality"}

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
            st.metric(f"{horizon}-Year Absolute Risk Reduction", f"{100*risk_reduction:.1f}%",
                     delta=f"Current: {100*r['baseline']:.1f}% → Target: {100*r['target']:.1f}%")
        if outcome == "mortality":
            st.caption("All-cause mortality includes deaths from any cause, including cancer and other diseases, not only cardiovascular disease.")

st.divider()

# 累積リスク曲線セクション
st.markdown("### 📈 Cumulative Risk Curves")

# 1. 全死亡の累積リスク曲線（信頼区間付き）
st.markdown("#### 💀 Cumulative Risk of All-Cause Mortality (95% CI)")
fig_mortality_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_mo_t = np.array(cumulative_data['mortality']['time'], dtype=float)
_mo_b = np.array(cumulative_data['mortality']['baseline_cumulative'], dtype=float)
_mo_tg = np.array(cumulative_data['mortality']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_mo_t, cutoff_year, side='right'))

fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[:cut_idx], y=_mo_b[:cut_idx], mode='lines', name='Current risk factors',
    line=dict(color='#ef5350', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[cut_idx:], y=_mo_b[cut_idx:], mode='lines', name='Current risk factors (estimated range: age ≥85)',
    line=dict(color='rgba(239,83,80,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[:cut_idx], y=_mo_tg[:cut_idx], mode='lines', name='At target',
    line=dict(color='#26a69a', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_mortality_cumulative.add_trace(go.Scatter(
    x=_mo_t[cut_idx:], y=_mo_tg[cut_idx:], mode='lines', name='At target (estimated range: age ≥85)',
    line=dict(color='rgba(38,166,154,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
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
    name='Current risk factors 95% CI',
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
    name='At target 95% CI',
    fillcolor='rgba(38,166,154,0.2)'
))

fig_mortality_cumulative.update_layout(
    title="Cumulative Risk of All-Cause Mortality (95% CI)",
    xaxis_title="Years",
    yaxis_title="Cumulative risk (%)",
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
st.caption("All-cause mortality includes deaths from any cause, including cancer and other diseases, not only cardiovascular disease.")


# 2. 心筋梗塞の累積リスク曲線（信頼区間付き）
st.markdown("#### 🫀 Cumulative Risk of Myocardial Infarction (95% CI)")
fig_mi_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_mi_t = np.array(cumulative_data['mi']['time'], dtype=float)
_mi_b = np.array(cumulative_data['mi']['baseline_cumulative'], dtype=float)
_mi_tg = np.array(cumulative_data['mi']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_mi_t, cutoff_year, side='right'))

# baseline: ～85歳
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[:cut_idx], y=_mi_b[:cut_idx], mode='lines', name='Current risk factors',
    line=dict(color='#ff6b6b', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
# baseline: 85歳～（薄色）
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[cut_idx:], y=_mi_b[cut_idx:], mode='lines', name='Current risk factors (estimated range: age ≥85)',
    line=dict(color='rgba(255,107,107,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
# target: ～85歳
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[:cut_idx], y=_mi_tg[:cut_idx], mode='lines', name='At target',
    line=dict(color='#4ecdc4', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
# target: 85歳～（薄色）
fig_mi_cumulative.add_trace(go.Scatter(
    x=_mi_t[cut_idx:], y=_mi_tg[cut_idx:], mode='lines', name='At target (estimated range: age ≥85)',
    line=dict(color='rgba(78,205,196,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
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
    name='Current risk factors 95% CI',
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
    name='At target 95% CI',
    fillcolor='rgba(78,205,196,0.2)'
))

fig_mi_cumulative.update_layout(
    title="Cumulative Risk of Myocardial Infarction (95% CI)",
    xaxis_title="Years",
    yaxis_title="Cumulative risk (%)",
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
st.markdown("#### 🧠 Cumulative Risk of Stroke (95% CI)")
fig_stroke_cumulative = go.Figure()

# 点推定値の線（85歳以上を薄色表示）
_st_t = np.array(cumulative_data['stroke']['time'], dtype=float)
_st_b = np.array(cumulative_data['stroke']['baseline_cumulative'], dtype=float)
_st_tg = np.array(cumulative_data['stroke']['target_cumulative'], dtype=float)
cutoff_year = max(0.0, 85.0 - float(age))
cut_idx = int(np.searchsorted(_st_t, cutoff_year, side='right'))

fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[:cut_idx], y=_st_b[:cut_idx], mode='lines', name='Current risk factors',
    line=dict(color='#ffa726', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[cut_idx:], y=_st_b[cut_idx:], mode='lines', name='Current risk factors (estimated range: age ≥85)',
    line=dict(color='rgba(255,167,38,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[:cut_idx], y=_st_tg[:cut_idx], mode='lines', name='At target',
    line=dict(color='#66bb6a', width=3), showlegend=True,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
))
fig_stroke_cumulative.add_trace(go.Scatter(
    x=_st_t[cut_idx:], y=_st_tg[cut_idx:], mode='lines', name='At target (estimated range: age ≥85)',
    line=dict(color='rgba(102,187,106,0.45)', width=3), showlegend=False,
    hovertemplate='%{x:.1f} years: %{y:.2f}%<extra></extra>'
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
    name='Current risk factors 95% CI',
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
    name='At target 95% CI',
    fillcolor='rgba(102,187,106,0.2)'
))

fig_stroke_cumulative.update_layout(
    title="Cumulative Risk of Stroke (95% CI)",
    xaxis_title="Years",
    yaxis_title="Cumulative risk (%)",
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
with st.expander("Sources and Notes"):
    st.markdown("""
**Baseline data (Japan):**
- Myocardial infarction: Miyagi AMI Registry (approximate 2014 values included in the CSV; replace when exact values become available)
- Stroke: Shiga Stroke Registry (representative values in the CSV)
- Mortality: Japanese 2023 life-table `qx` values are **used directly from the CSV `qx` column** (recommended). A temporary Gompertz approximation is used if the CSV is unavailable.
    """)

# 一次予防モデル脚注（整形済み）
with st.expander("Primary Prevention Model Notes (Myocardial Infarction, Stroke, and All-Cause Mortality)"):
    st.markdown("""
**Objective**: Visualize differences in the cumulative risk of major outcomes (MI, stroke, and all-cause mortality) associated with improving routinely available outpatient risk factors (SBP, LDL-C, HbA1c, smoking, BMI, and CKD [eGFR/albuminuria]).

**Baseline incidence**: Interpolated from age- and sex-specific CSV data (life-table `qx` for mortality; annual MI and stroke rates converted to probabilities).

**Cumulative calculation (discrete time)**:
- Update age each year to the age at year t
- Annual event probability: q_t = baseline(age_t, sex, outcome) × RR_total(age_t)
- Cumulative risk: CumRisk_{t+1} = CumRisk_t + (1 − CumRisk_t) × q_t

**Age attenuation**: Because relative effects tend to diminish at older ages, each factor's ln(RR) is adjusted using a coefficient, α(age). Ages ≥85 are treated conservatively as an estimated range (weaker or near-zero relative effects).

1) SBP (systolic blood pressure)
- Unit effect: HR ≈0.91 per 5 mmHg reduction (stronger for stroke and slightly weaker for MI)
- Age attenuation: α_SBP(age)=1.0 (≤75) → linear decline → 0.0 (85) → 0.0 thereafter

2) LDL-C
- Unit effect: HR ≈0.77 per 1 mmol/L (≈38.7 mg/dL) reduction
- Age attenuation: α_LDL(age)=1.0 (≤85) → 0.7 (90) → 0.7 thereafter (mild attenuation)

3) HbA1c (macrovascular outcomes)
- Direction of effect: a 1% reduction produces a modest RR <1. Age attenuation: α_A1c(age)=1.0 (≤75) → 0.0 (85) → 0.0 thereafter
- Microvascular outcomes are outside the scope of this model (potential future extension).

4) Smoking
- Current smoking increases the HR. After cessation, it declines according to HR(y)=1+(HR0−1)×exp(−k·y) (k≈0.15–0.2).
- Age attenuation is mild (relative differences persist across a broad age range).

5) BMI (U-shaped relationship with age shift)
- The optimal BMI (nadir) shifts with age from 23.5 to 26.5 (ages 40 to 80).
- High BMI: stronger effect at younger ages and neutralized at very advanced ages. Low BMI: more adverse at older ages.
- Continuous model per BMI unit: β=ln(RR5)/5, RR=exp(β×ΔBMI). Re-evaluated and multiplied annually. Extreme inputs are clipped to 0.5–2.0.

6) CKD (eGFR/albuminuria)
- Point-estimate RRs: eGFR ≥60=1.0, 45–59=1.30, <45=1.80. ACR: A1=1.0, A2=1.35, A3=1.90.
- Initially combined using max(rr_eGFR, rr_ACR) (advanced settings may permit multiplication with a cap).
- Age attenuation (lnRR×α): A2/A3: 1.0 → 0.85 (85) → 0.80 (>85); low eGFR alone: 1.0 → 0.80 → 0.70; both present: 1.0 → 0.90 → 0.85.
- Outcome-specific adjustment: MI×0.8 / stroke×1.0 / mortality×1.1.

**Output notes**
- Evidence range: Relative effects are generally supported through approximately age 85. Values above age 85 are conservatively adjusted as estimates (lines for ages ≥85 are shown in lighter colors).
- Avoiding double counting: Effects such as sodium reduction are represented through SBP. BMI and waist circumference should not both be strongly weighted.
- Absolute vs relative effects: Relative differences may narrow at older ages, while absolute risk reduction (ARR) may be maintained or increase.

**Representative references (examples)**
- Blood pressure: SPRINT (NEJM 2015), HYVET (NEJM 2008), BPLTTC / Rahimi et al. (Lancet 2021)
- Lipids: CTT meta-analyses (Lancet series), WOSCOPS (NEJM 1995), ASCOT-LLA (Lancet 2003), JUPITER (NEJM 2008)
- Glycemia: UKPDS, ADVANCE (NEJM 2008), ACCORD (NEJM 2008/2010), Selvin et al. (Diabetes Care)
- Smoking: INTERHEART (Lancet 2004), national cohort studies/public health reports
- BMI: Prospective Studies Collaboration (Lancet 2009)
- CKD: CKD-PC (Lancet 2010/2012 and others), HOPE/RENAAL/IDNT (NEJM 2000–2001), SPRINT/HYVET
    """)
