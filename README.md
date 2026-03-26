# JP Outcomes Prevention Simulator (MVP)

主要アウトカム（MI/Stroke/All-cause mortality）の累積リスク（%）を、外来で取得できる因子（SBP, LDL-C, HbA1c, 喫煙, BMI, CKD）で年次に評価・可視化する Streamlit アプリです。95%CI、年齢減衰、85歳以上の推定域表示に対応。
URL：https://japan-cvd-risk-simulator.streamlit.app/

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit_outcomes.py
```

ブラウザ: http://localhost:8501

## Requirements

```
streamlit
plotly
numpy
pandas
pyyaml
scipy
```

## Data (CSV)

CSV は **`data/` 配下またはリポジトリ直下**のどちらでも配置可（自動フォールバック）。
- `baseline_incidence_mi.csv` — Miyagi AMI approximated points.
- `baseline_incidence_stroke.csv` — Shiga Stroke Registry points.
- `baseline_incidence_mortality.csv` — 推奨: 令和5年簡易生命表 `qx` 列を格納（0–1）。無い場合は `incidence_per_100k` を 1/100000 に換算。

## Config

`config.yaml` をアプリ直下に配置。
- `baseline_incidence`: CSV無時の近似 a,b
- `baseline_ci_percent`: ベースライン確率のCI幅（±p）
- `risk_models`: SBP/LDL/HbA1c/Smoking の係数
- `age_effect`: 75–85歳等での年齢減衰設定

## How it works (brief)

- 年次離散累積: `q_t = baseline(age_t,sex,outcome) × RR_total(age_t)`、`CumRisk_{t+1} = CumRisk_t + (1−CumRisk_t)×q_t`
- RR: SBP, LDL-C, HbA1c（macro控えめ）, 喫煙（禁煙で指数減衰）, BMI（U字＋年齢シフト）, CKD（eGFR/ACR, 年齢減衰・アウトカム別調整）
- 95%CI: 効果量（target側）＋ベースライン確率（±p）
- 85歳以上: 推定域としてメイン線を薄色表示

## Usage

1) 左サイドバーに現在→目標を入力（任意で BMI/CKD）。
2) 期間（最大50年）を選択し「🔄 リスク計算を実行」。
3) サマリーと各アウトカムのグラフ（累積%・95%CI）を確認。

## Notes

- 本ツールは教育・共有意思決定のためのMVPで、医療機器ではありません。
- 効果量と年齢減衰は代表的エビデンスに基づく近似であり、患者個別の因果効果を保証しません。
- ベースラインの国・年代依存性に留意（CSV差し替えで調整可）。

## References (selected formal citations)

- Rahimi K, et al. The Lancet. 2021;398(10305):1053–1064.
- Beckett NS, et al. New England Journal of Medicine. 2008;358(18):1887–1898. (HYVET)
- SPRINT Research Group. New England Journal of Medicine. 2015;373(22):2103–2116.
- Andersson C, et al. The Lancet Healthy Longevity. 2023;4(3):e180–e191.
- CTT Collaboration. The Lancet. 2019;393(10170):407–415.
- Sattar N, et al. European Heart Journal. 2021;42(15):1507–1516.
- Wan EYF, et al. Diabetes Care. 2020;43(8):1742–1750.
- ADVANCE Collaborative Group. New England Journal of Medicine. 2008;358(24):2560–2572.
- ACCORD Study Group. New England Journal of Medicine. 2008;358(24):2545–2559.
- Di Angelantonio E, et al. The Lancet. 2016;388(10046):776–786.
- Flegal KM, et al. JAMA. 2013;309(1):71–82.
- Zheng W, et al. New England Journal of Medicine. 2011;364(9):829–841.
- GBD 2019 Tobacco Collaborators. The Lancet. 2021;397(10292):2337–2360.
- Doll R, et al. BMJ. 2004;328(7455):1519–1528.
- Matsushita K, et al. The Lancet. 2020;395(10225):709–718.

## モデル構造とエビデンスサマリー（追記）

本モデルは、SBP/LDL-C/HbA1c/喫煙/BMI/CKD について年齢による効果減衰と年次累積を統合したシミュレーターです。Framingham/ASCVD などの静的スコアと異なり、加齢による効果逓減、疾患特異的補正、逐次積算を実装しています。

### 基本構造
- 各因子 HR(age) を逐次適用して年間リスクを積算
- 加齢で相対効果は減弱（特に ≥85 歳でほぼ消失）
- ベースライン発症率（生命表/レジストリ）に修正係数を乗じて推定

### 各因子（要旨）
1) SBP: 5 mmHg 低下ごと HR≈0.91（55–84 歳）。≥85 歳は HR≈0.99（Rahimi 2021, HYVET, SPRINT）
2) LDL-C: 1 mmol/L 低下ごと HR≈0.77。≥85 歳で緩やかに減弱（Andersson 2023, CTT 2019, Sattar 2021）
3) HbA1c: 1% 上昇 HR≈1.14（中年〜前期高齢）。高齢で HR≈1.08–1.10（Wan 2020, ADVANCE, ACCORD）
4) BMI: U 字＋年齢シフト（谷 23.5→26.5）。高 BMI は若年で強く、高齢で中立化。低 BMI は高齢で不利（Di Angelantonio 2016 ほか）
5) 喫煙: 現喫煙 HR↑、禁煙で指数的低下（GBD 2021, Doll 2004）
6) CKD: eGFR/ACR 異常で HR↑。MI×0.8, Stroke×1.0, Mortality×1.1 の補正（Matsushita 2020 ほか）

### 年齢補正の総合適用（例）

| 年齢帯 | 降圧 | LDL | HbA1c | BMI | 喫煙 | CKD |
|---|---|---|---|---|---|---|
| <65 | 強い | 強い | 明確 | 有意(U字強) | 強 | 強 |
| 65–74 | 中等度 | 有効 | 維持 | 緩やか | 強 | 強 |
| 75–84 | 軽度 | 有効 | やや減弱 | 平坦化 | 強 | 強 |
| ≥85 | ほぼ消失 | 弱い | フラット | 反転傾向 | やや希釈 | 維持 |

### 累積推定
- baseline × ∏HR(age) を年次で逐次積算。若年で介入効果大、晩年は限定的（ARR は維持/増大し得る）
