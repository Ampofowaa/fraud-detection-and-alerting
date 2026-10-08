"""Fraud alert dashboard for a fraud operations team.

Reads aggregated results exported by 02_ieee_cis_fraud.ipynb (dashboard/data/results.json).
No row-level transaction data is used.

Run locally:  streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent / 'data' / 'results.json'

# Categorical slots (fixed order) and reserved status colours
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
GOOD, WARNING, CRITICAL = '#0ca30c', '#fab219', '#d03b3b'

st.set_page_config(page_title = 'Fraud Alert Dashboard', page_icon = '🛡️', layout = 'wide')


@st.cache_data
def load_results():
    return json.loads(RESULTS.read_text())


r = load_results()
totals = r['totals']
curve = pd.DataFrame(r['capture_curve'])

# --- Header and business question ------------------------------------------------
st.title('Fraud Alert Dashboard')
st.markdown(
    "**Business question:** a fraud operations team can only review a limited number of transactions each day. "
    "*Which transactions should it review to minimise fraud losses plus review costs, and how will it know when "
    "the model needs retraining?*"
)
st.caption(
    f"Results on an unseen test month: {totals['test_days']} days, {totals['test_transactions']:,} real e-commerce "
    f"transactions (IEEE-CIS), {totals['test_frauds']:,} frauds worth ${totals['test_fraud_value']:,.0f}. "
    f"Model: {r['final_model']}."
)

# --- Controls ----------------------------------------------------------------------
with st.sidebar:
    st.header('Operating assumptions')
    review_cost = st.slider('Cost of reviewing one alert ($)', 1, 50, 5,
                            help = 'Analyst time per alert. The recommended threshold is re-chosen on the validation month for each value.')
    mode = st.radio('Alert threshold', ['Recommended (cost-optimal)', 'Custom alert volume'])
    custom_rate = st.slider('Alerts per 1,000 transactions', 1, 300, 30) if mode == 'Custom alert volume' else None
    st.divider()
    st.caption('The recommended threshold is chosen on the validation month and applied unchanged to the test month, '
               'so the test results are an honest out-of-time estimate.')

curve['review_cost'] = review_cost * curve['alerts']
curve['missed_value'] = totals['test_fraud_value'] - curve['fraud_value_caught']
curve['total_cost'] = curve['missed_value'] + curve['review_cost']

if custom_rate is None:
    point = r['recommended'][str(review_cost)]
else:
    nearest = int((curve['alerts_per_1000'] - custom_rate).abs().to_numpy().argmin())
    point = curve.iloc[nearest].to_dict()

alerts = point['alerts']
caught_value = point['fraud_value_caught']
total_cost = (totals['test_fraud_value'] - caught_value) + review_cost * alerts
saving = totals['test_fraud_value'] - total_cost

# --- KPI tiles ---------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('Fraud value caught', f"${caught_value:,.0f}", f"{100 * caught_value / totals['test_fraud_value']:.1f}% of fraud losses",
          delta_color = 'off')
k2.metric('Net saving vs no model', f"${saving:,.0f}", f"{100 * saving / totals['test_fraud_value']:.1f}% of fraud losses",
          delta_color = 'normal' if saving >= 0 else 'inverse')
k3.metric('Alerts per day', f"{alerts / totals['test_days']:,.0f}", f"{point['alerts_per_1000']:.1f} per 1,000 transactions",
          delta_color = 'off')
k4.metric('Alert precision', f"{point['frauds_caught'] / max(alerts, 1):.1%}", 'of alerts are fraud', delta_color = 'off')
k5.metric('Fraud cases caught', f"{point['frauds_caught']:,}", f"{100 * point['frauds_caught'] / totals['test_frauds']:.1f}% of cases",
          delta_color = 'off')

tab_ops, tab_model, tab_monitor, tab_about = st.tabs(['Operating point', 'Model performance', 'Monitoring', 'About'])

# --- Tab 1: operating point -------------------------------------------------------
with tab_ops:
    chosen = pd.DataFrame([{'alerts_per_1000': point['alerts_per_1000'], 'total_cost': total_cost,
                            'value_caught_pct': 100 * caught_value / totals['test_fraud_value']}])
    curve['value_caught_pct'] = 100 * curve['fraud_value_caught'] / totals['test_fraud_value']

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Total cost vs alert volume')
        st.caption('Missed fraud value + review cost on the test month. The dot is the selected threshold.')
        base = alt.Chart(curve).mark_line(color = SERIES[0], strokeWidth = 2).encode(
            x = alt.X('alerts_per_1000:Q', title = 'Alerts per 1,000 transactions'),
            y = alt.Y('total_cost:Q', title = 'Total cost ($)', axis = alt.Axis(format = '$,.0f')),
            tooltip = [alt.Tooltip('alerts_per_1000:Q', title = 'Alerts per 1,000', format = '.1f'),
                       alt.Tooltip('total_cost:Q', title = 'Total cost', format = '$,.0f')],
        )
        dot = alt.Chart(chosen).mark_point(filled = True, size = 120, color = SERIES[1], stroke = 'white', strokeWidth = 2).encode(
            x = 'alerts_per_1000:Q', y = 'total_cost:Q')
        st.altair_chart(base + dot, width = 'stretch')
    with c2:
        st.subheader('Fraud value caught vs alert volume')
        st.caption('Diminishing returns: each extra alert catches less fraud value.')
        base = alt.Chart(curve).mark_line(color = SERIES[0], strokeWidth = 2).encode(
            x = alt.X('alerts_per_1000:Q', title = 'Alerts per 1,000 transactions'),
            y = alt.Y('value_caught_pct:Q', title = 'Fraud value caught (%)', scale = alt.Scale(domain = [0, 100])),
            tooltip = [alt.Tooltip('alerts_per_1000:Q', title = 'Alerts per 1,000', format = '.1f'),
                       alt.Tooltip('value_caught_pct:Q', title = 'Fraud value caught (%)', format = '.1f')],
        )
        dot = alt.Chart(chosen).mark_point(filled = True, size = 120, color = SERIES[1], stroke = 'white', strokeWidth = 2).encode(
            x = 'alerts_per_1000:Q', y = 'value_caught_pct:Q')
        st.altair_chart(base + dot, width = 'stretch')

    st.subheader('How the recommended threshold responds to review cost')
    sens = pd.DataFrame([
        {'Review cost ($)': c,
         'Alerts per 1,000': v['alerts_per_1000'],
         'Alert precision (%)': 100 * v['frauds_caught'] / max(v['alerts'], 1),
         'Fraud value caught (%)': 100 * v['fraud_value_caught'] / totals['test_fraud_value']}
        for c, v in ((int(c), v) for c, v in r['recommended'].items()) if c in (1, 5, 10, 25, 50)
    ]).set_index('Review cost ($)')
    st.dataframe(sens.style.format('{:.1f}'), width = 'stretch')
    st.caption('Cheaper reviews justify more alerts. Expensive reviews push the team to focus on the highest-risk transactions.')

# --- Tab 2: model performance ------------------------------------------------------
with tab_model:
    st.subheader('Model comparison')
    st.caption(f"Average Precision is the primary metric because only {totals['test_fraud_rate']:.1%} of transactions are fraud. "
               f"Lift = AP ÷ fraud rate (a random model scores 1x).")
    test_tbl = pd.DataFrame(r['test_results']).T
    valid_tbl = pd.DataFrame(r['validation_results']).T
    comparison = pd.DataFrame({
        'Validation AP': valid_tbl['Average Precision'],
        'Test AP': test_tbl['Average Precision'],
        'Test lift': test_tbl['lift over base rate'],
        'Test ROC-AUC': test_tbl['ROC-AUC'],
    })
    st.dataframe(comparison.style.format({'Validation AP': '{:.3f}', 'Test AP': '{:.3f}', 'Test lift': '{:.1f}x',
                                          'Test ROC-AUC': '{:.3f}'}, na_rep = '-'), width = 'stretch')
    st.caption(f"Final model test AP 95% bootstrap interval: {r['test_ap_ci']}. "
               "Results are not comparable with the Kaggle leaderboard (different split; ranked on ROC-AUC).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Precision-recall on the test month')
        order = [r['final_model']] + [m for m in r['pr_curves'] if m != r['final_model']]
        pr = pd.concat([pd.DataFrame({**r['pr_curves'][m], 'model': m}) for m in order])
        chart = alt.Chart(pr).mark_line(strokeWidth = 2).encode(
            x = alt.X('recall:Q', title = 'Recall (share of fraud caught)'),
            y = alt.Y('precision:Q', title = 'Precision (share of alerts that are fraud)', scale = alt.Scale(domain = [0, 1])),
            color = alt.Color('model:N', scale = alt.Scale(domain = order, range = SERIES[:len(order)]),
                              legend = alt.Legend(orient = 'bottom', title = None, direction = 'vertical')),
            tooltip = ['model:N', alt.Tooltip('recall:Q', format = '.2f'), alt.Tooltip('precision:Q', format = '.2f')],
        )
        st.altair_chart(chart, width = 'stretch')
    with c2:
        st.subheader('What drives the model')
        imp = pd.Series(r['feature_importance']).rename_axis('feature').reset_index(name = 'gain')
        chart = alt.Chart(imp).mark_bar(color = SERIES[0], cornerRadiusEnd = 4).encode(
            x = alt.X('gain:Q', title = 'Total gain'),
            y = alt.Y('feature:N', sort = '-x', title = None),
            tooltip = ['feature:N', alt.Tooltip('gain:Q', format = ',.0f')],
        )
        st.altair_chart(chart, width = 'stretch')
        st.caption('Top 15 features by LightGBM gain. Many raw columns are anonymised by the data provider.')

# --- Tab 3: monitoring ------------------------------------------------------------
with tab_monitor:
    st.markdown(
        "**Why label-free monitoring:** fraud labels arrive weeks late, usually when a cardholder disputes a charge. "
        "PSI on the model score needs no labels, so it is the **early warning**. Average Precision per period is the "
        "**lagging confirmation** once labels mature."
    )
    perf = pd.DataFrame(r['period_performance']).T
    perf.index = perf.index.astype(int)
    perf = perf.rename_axis('period').reset_index()
    numeric_cols = ['fraud rate', 'Average Precision', 'score PSI', 'lift']
    perf[numeric_cols] = perf[numeric_cols].astype(float)
    perf['status'] = pd.cut(perf['score PSI'], [-1, 0.1, 0.25, 99], labels = ['Stable', 'Moderate shift', 'Significant shift'])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Score drift (PSI) by period')
        st.caption('Reference: training window. Below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant.')
        chart = alt.Chart(perf).mark_bar(cornerRadiusEnd = 4).encode(
            x = alt.X('period:O', title = 'Period (30-day blocks)'),
            y = alt.Y('score PSI:Q', title = 'PSI'),
            color = alt.Color('status:N', scale = alt.Scale(domain = ['Stable', 'Moderate shift', 'Significant shift'],
                                                            range = [GOOD, WARNING, CRITICAL]),
                              legend = alt.Legend(orient = 'bottom', title = None)),
            tooltip = ['period:O', alt.Tooltip('score PSI:Q', format = '.3f'), 'status:N', 'used for:N'],
        )
        st.altair_chart(chart, width = 'stretch')
    with c2:
        st.subheader('Average Precision by period')
        st.caption('Periods used for training are in-sample, so their AP is optimistic.')
        chart = alt.Chart(perf).mark_line(color = SERIES[0], strokeWidth = 2, point = alt.OverlayMarkDef(size = 80)).encode(
            x = alt.X('period:O', title = 'Period (30-day blocks)'),
            y = alt.Y('Average Precision:Q', scale = alt.Scale(domain = [0, 1])),
            tooltip = ['period:O', alt.Tooltip('Average Precision:Q', format = '.3f'),
                       alt.Tooltip('fraud rate:Q', format = '.2%'), 'used for:N'],
        )
        st.altair_chart(chart, width = 'stretch')

    st.subheader('Feature drift (PSI vs training window)')
    psi = pd.DataFrame(r['psi'])
    st.dataframe(psi.T.style.format('{:.3f}').background_gradient(cmap = 'Blues', vmin = 0, vmax = 0.25),
                 width = 'stretch')
    st.caption('Rows: the model score and the 10 most important features. Darker = larger shift.')

# --- Tab 4: about -----------------------------------------------------------------
with tab_about:
    mem = r['memory_report']
    st.markdown(f"""
**Data.** IEEE-CIS Fraud Detection (Vesta Corporation, via Kaggle): 590,540 real e-commerce transactions x 434 features,
about 6 months, 3.5% fraud. Loaded with compact dtypes and cached as Parquet: memory **{mem.get('naive memory (GB)', float('nan')):.2f} GB →
{mem['optimised memory (GB)']:.2f} GB**.

**Method.**
- Time-based split: train on the earliest ~4 months, validate on the next month, test once on the final month.
- LightGBM, compared against logistic regression and IsolationForest baselines, with Average Precision as the primary metric.
- Per-card history features computed from earlier transactions only, so there is no leakage.
- Alert threshold chosen on validation to minimise missed fraud value + review cost.
- Drift monitored with the Population Stability Index (PSI).

**Limitations.** Many columns are anonymised. The data comes from one e-commerce merchant over ~6 months. Labels come from chargebacks,
so some fraud is likely labelled as normal. Review cost is an assumption, and customer friction is not costed.

**Data sharing.** This dashboard uses only aggregated results. No row-level transaction data is published.
""")
