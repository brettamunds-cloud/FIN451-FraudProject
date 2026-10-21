"""
app.py - Interactive fraud & transaction monitoring dashboard (Streamlit).

Run it with:   streamlit run app.py

What the screen shows, top to bottom:
  * Sidebar: choose the data (built-in synthetic data or your own CSV), then tune
    the risk threshold and how many points each rule is worth.
  * Summary numbers for the dataset.
  * Detection performance: how many planted anomalies we caught vs. missed
    (only when the data has an injected_anomaly answer-key column).
  * Charts.
  * The alert queue: every flagged transaction with a plain-English explanation.

All the actual fraud logic lives in scoring.py - this file is just the user interface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import generate_data
import scoring

DATA_FILE = Path(__file__).parent / generate_data.OUTPUT_FILE

# Chart colors: gray for ordinary activity, red reserved for alerts,
# blue/orange for caught vs. missed.
COLOR_NORMAL = "#9c9a92"
COLOR_FLAGGED = "#e34948"
COLOR_CAUGHT = "#2a78d6"
COLOR_MISSED = "#eb6834"

st.set_page_config(page_title="FIN451 Fraud Monitor", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached so it only re-runs when the data itself changes)
# ---------------------------------------------------------------------------
def data_file_version():
    """Last-modified time of transactions.csv, so the cache refreshes if you re-run generate_data.py."""
    return DATA_FILE.stat().st_mtime if DATA_FILE.exists() else None


@st.cache_data(show_spinner="Preparing data and running rules...")
def prepare_data(raw_bytes=None, file_version=None):
    """Read, validate, and run the rules once. Returns (signals_df, warnings).

    raw_bytes=None means "use the built-in synthetic dataset" (generated on the fly
    if transactions.csv doesn't exist yet). file_version is only there so Streamlit's
    cache notices when that file changes.
    """
    if raw_bytes is None:
        if not DATA_FILE.exists():
            generate_data.main(output_path=DATA_FILE)
        raw = scoring.read_transactions_csv(DATA_FILE)
    else:
        raw = scoring.read_transactions_csv(raw_bytes)
    clean, warnings = scoring.load_and_validate(raw)
    return scoring.compute_rule_signals(clean), warnings


def show_expected_format():
    """Show users what a valid file looks like after an upload error."""
    st.markdown("**Expected CSV format** (column names are case-insensitive; common "
                "alternatives like `Amt` or `Date` are recognised):")
    st.dataframe(pd.DataFrame({
        "transaction_id": ["TXN00001", "TXN00002"],
        "customer_id": ["C001", "C001"],
        "timestamp": ["2026-07-01 14:05:00", "2026-07-01 18:30:00"],
        "amount": [42.50, 9500.00],
        "transaction_type": ["card purchase", "cash deposit"],
        "merchant": ["Target", "Chase Branch Deposit"],
        "location": ["Chicago, IL", "Chicago, IL"],
    }), hide_index=True)
    st.caption("Required: customer_id, timestamp, amount. Optional: transaction_id, "
               "transaction_type, merchant, location, injected_anomaly (True/False answer key).")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def reset_controls():
    """Put the threshold and all weights back to the defaults in scoring.py."""
    st.session_state["threshold"] = scoring.DEFAULT_THRESHOLD
    for rule, weight in scoring.DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{rule}"] = weight


RULE_HELP = {
    "amount": "Points for amounts far above this customer's own typical spending "
              f"(partial from z-score {scoring.Z_FIRE:g}, full at {scoring.Z_FULL:g}).",
    "velocity": f"Points when {scoring.VELOCITY_MIN_COUNT}+ transactions happen within one hour.",
    "late_night": "Points for activity between 12:00 and 4:59 AM.",
    "new_place": "Full points for a brand-new city AND merchant; half for just one.",
    "structuring": "Points for cash deposits of $9,000-$9,999.99 (just under the $10,000 CTR line).",
}


def sidebar_controls():
    """Draw the sidebar and return (uploaded_bytes_or_None, weights, threshold)."""
    if "threshold" not in st.session_state:
        reset_controls()

    st.sidebar.header("1. Data")
    source = st.sidebar.radio("Data source", ["Built-in synthetic data", "Upload my own CSV"])
    uploaded_bytes = None
    if source == "Upload my own CSV":
        upload = st.sidebar.file_uploader("Transactions CSV", type=["csv"])
        if upload is None:
            st.info("Upload a CSV in the sidebar to begin, or switch back to the built-in data.")
            show_expected_format()
            st.stop()
        uploaded_bytes = upload.getvalue()

    st.sidebar.header("2. Alert threshold")
    threshold = st.sidebar.slider(
        "Flag transactions scoring at least", 0, 100, key="threshold",
        help="Lower = more alerts (catch more fraud, but more false alarms for analysts to clear). "
             "Higher = fewer alerts (less work, but more fraud slips through).")

    st.sidebar.header("3. Rule weights (max points)")
    weights = {}
    for rule in scoring.RULES:
        weights[rule] = st.sidebar.slider(scoring.RULE_LABELS[rule], 0, 100,
                                          key=f"w_{rule}", help=RULE_HELP[rule])
    st.sidebar.button("Reset to defaults", on_click=reset_controls)
    return uploaded_bytes, weights, threshold


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def show_summary(scored):
    """Headline numbers about the dataset and the alerts."""
    n_flagged = int(scored["flagged"].sum())
    c1, c2, c3, c4, c5 = st.columns([1, 0.8, 1.8, 1.3, 1.2])
    c1.metric("Transactions", f"{len(scored):,}")
    c2.metric("Customers", f"{scored['customer_id'].nunique():,}")
    c3.metric("Date range", f"{scored['timestamp'].min():%b %d} - {scored['timestamp'].max():%b %d}",
              help=f"{scored['timestamp'].min():%B %d, %Y} to {scored['timestamp'].max():%B %d, %Y}")
    c4.metric("Total volume", f"${scored['amount'].sum():,.0f}")
    c5.metric("Flagged for review", f"{n_flagged:,}", f"{n_flagged / len(scored):.1%} of transactions",
              delta_color="off")


def show_performance(scored):
    """Caught vs. missed against the injected_anomaly answer key."""
    result = scoring.evaluate_detection(scored)
    st.subheader("Detection performance vs. planted anomalies")
    if result is None:
        st.caption("This file has no `injected_anomaly` column, so there's no answer key to grade against.")
        return
    if result["injected"] == 0:
        st.caption("The answer-key column has no planted anomalies marked True.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Planted anomalies", result["injected"])
    c2.metric("Caught", result["caught"])
    c3.metric("Missed", result["missed"])
    c4.metric("Recall (catch rate)", f"{result['recall']:.0%}",
              help="Of all planted anomalies, the share we flagged.")
    precision = result["precision"]
    c5.metric("Precision", "n/a" if np.isnan(precision) else f"{precision:.0%}",
              help=f"Of all {result['flagged']} flags, the share that were real anomalies. "
                   f"The other {result['false_alarms']} are false alarms an analyst would clear.")

    st.caption(
        "**Missed** anomalies are false negatives: fraud or money laundering that got through, "
        "which risks losses and regulatory penalties. **False alarms** are false positives: "
        "honest customers whose transactions an analyst must review, which costs staff time and "
        "can annoy customers. Moving the threshold trades one for the other.")

    if result["by_type"] is not None:
        by_type = result["by_type"]
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(chart_caught_by_type(by_type), width="stretch")
        with right:
            st.dataframe(by_type.rename(columns={"anomaly_type": "Anomaly type", "caught": "Caught",
                                                 "injected": "Planted", "missed": "Missed",
                                                 "catch_rate": "Catch rate"}),
                         hide_index=True, width="stretch",
                         column_config={"Catch rate": st.column_config.ProgressColumn(
                             format="percent", min_value=0, max_value=1)})
        missed = scored[scored["injected_anomaly"] & ~scored["flagged"]]
        if len(missed):
            with st.expander(f"Inspect the {len(missed)} missed anomalies"):
                st.dataframe(missed[["transaction_id", "customer_id", "timestamp", "amount",
                                     "transaction_type", "merchant", "location", "anomaly_type",
                                     "risk_score", "explanation"]],
                             hide_index=True, width="stretch")


def chart_amount_distribution(scored):
    """Chart 1: how big are transactions? Log scale, because amounts span $1 to $10,000+."""
    data = scored.assign(log_amount=np.log10(scored["amount"].clip(lower=1)),
                         status=np.where(scored["flagged"], "Flagged", "Not flagged"))
    fig = px.histogram(data, x="log_amount", color="status", nbins=60, barmode="stack",
                       color_discrete_map={"Not flagged": COLOR_NORMAL, "Flagged": COLOR_FLAGGED},
                       category_orders={"status": ["Not flagged", "Flagged"]},
                       labels={"log_amount": "Transaction amount (log scale)", "status": ""})
    ticks = [0, 1, 2, 3, 4]
    fig.update_xaxes(tickvals=ticks, ticktext=[f"${10 ** t:,}" for t in ticks])
    fig.update_traces(marker_line_width=0.5, marker_line_color="white",
                      hovertemplate="%{y} transactions<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Amount distribution", yaxis_title="Transactions",
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60))
    return fig


def chart_over_time(scored):
    """Chart 2: every transaction over time; flagged ones in red on top."""
    fig = go.Figure()
    for label, subset, color, size in [
        ("Not flagged", scored[~scored["flagged"]], COLOR_NORMAL, 5),
        ("Flagged", scored[scored["flagged"]], COLOR_FLAGGED, 8),
    ]:
        fig.add_trace(go.Scattergl(
            x=subset["timestamp"], y=subset["amount"], mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=0.5 if label == "Not flagged" else 0.9,
                        line=dict(width=1, color="white") if label == "Flagged" else None),
            customdata=subset[["transaction_id", "customer_id", "transaction_type", "risk_score"]],
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{x|%b %d %I:%M %p}<br>"
                          "$%{y:,.2f} %{customdata[2]}<br>Risk score %{customdata[3]}<extra></extra>"))
    fig.update_yaxes(type="log", dtick=1, title="Amount ($, log scale)", tickprefix="$")
    fig.update_layout(title="Transactions over time (flags highlighted)", xaxis_title=None,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60))
    return fig


def chart_risk_scores(scored, threshold):
    """Chart 3: risk score distribution, with the threshold drawn as a line."""
    data = scored.assign(status=np.where(scored["flagged"], "Flagged", "Not flagged"))
    fig = px.histogram(data, x="risk_score", color="status", nbins=50, barmode="stack",
                       color_discrete_map={"Not flagged": COLOR_NORMAL, "Flagged": COLOR_FLAGGED},
                       category_orders={"status": ["Not flagged", "Flagged"]},
                       labels={"risk_score": "Risk score (0-100)", "status": ""}, log_y=True)
    fig.add_vline(x=threshold, line_dash="dash", line_color="#52514e",
                  annotation_text=f"Threshold {threshold}", annotation_position="top right")
    fig.update_traces(marker_line_width=0.5, marker_line_color="white",
                      hovertemplate="Score %{x}: %{y} transactions<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Risk score distribution", yaxis_title="Transactions (log scale)",
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60))
    fig.update_xaxes(range=[-2, 102])
    fig.update_yaxes(dtick=1)          # label only 1, 10, 100, 1000 on the log axis
    return fig


def chart_flags_by_rule(scored):
    """Chart 4: among flagged transactions, how often did each rule contribute?"""
    flagged = scored[scored["flagged"]]
    counts = pd.DataFrame({
        "Rule": [scoring.RULE_LABELS[r] for r in scoring.RULES],
        "Flagged transactions": [int((flagged[f"pts_{r}"] > 0).sum()) for r in scoring.RULES],
    }).sort_values("Flagged transactions")
    fig = px.bar(counts, x="Flagged transactions", y="Rule", orientation="h", text="Flagged transactions")
    fig.update_traces(marker_color=COLOR_FLAGGED, textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x} flagged transactions<extra></extra>")
    fig.update_layout(title="Flags by rule type (a transaction can trip several rules)",
                      yaxis_title=None, margin=dict(t=60, r=40))
    return fig


def chart_caught_by_type(by_type):
    """Chart 5: caught vs. missed for each kind of planted anomaly."""
    data = by_type.melt(id_vars="anomaly_type", value_vars=["caught", "missed"],
                        var_name="Outcome", value_name="Count")
    data["Outcome"] = data["Outcome"].str.capitalize()
    fig = px.bar(data, x="Count", y="anomaly_type", color="Outcome", orientation="h",
                 color_discrete_map={"Caught": COLOR_CAUGHT, "Missed": COLOR_MISSED},
                 labels={"anomaly_type": ""})
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="%{y}: %{x}<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Caught vs. missed by anomaly type", legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(t=70))
    return fig


def show_charts(scored, threshold):
    st.subheader("Charts")
    left, right = st.columns(2)
    left.plotly_chart(chart_amount_distribution(scored), width="stretch")
    right.plotly_chart(chart_risk_scores(scored, threshold), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(chart_over_time(scored), width="stretch")
    right.plotly_chart(chart_flags_by_rule(scored), width="stretch")


def show_alert_queue(scored):
    """The analyst's to-do list: flagged transactions, riskiest first."""
    flagged = scored[scored["flagged"]].sort_values("risk_score", ascending=False)
    st.subheader(f"Alert queue: {len(flagged):,} flagged transactions")
    if flagged.empty:
        st.success("Nothing is above the threshold. Try lowering it in the sidebar.")
        return

    customers = st.multiselect("Filter by customer", sorted(flagged["customer_id"].unique()))
    if customers:
        flagged = flagged[flagged["customer_id"].isin(customers)]

    columns = ["risk_score", "transaction_id", "customer_id", "timestamp", "amount",
               "transaction_type", "merchant", "location", "rules_fired", "explanation"]
    if "injected_anomaly" in flagged.columns:
        columns.append("injected_anomaly")
    st.dataframe(
        flagged[columns], hide_index=True, width="stretch", height=420,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100,
                                                          format="%.0f"),
            "timestamp": st.column_config.DatetimeColumn("Time", format="MMM D, YYYY h:mm A"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "rules_fired": "Rules fired",
            "explanation": st.column_config.TextColumn("Explanation", width="large"),
            "injected_anomaly": "Planted?",
        })
    st.download_button("Download flagged transactions (CSV)",
                       flagged[columns].to_csv(index=False).encode("utf-8"),
                       file_name="flagged_transactions.csv", mime="text/csv")


def show_rules_explainer():
    with st.expander("How the risk score works"):
        st.markdown(f"""
Each rule adds up to its **weight** in points (set in the sidebar). The total is capped at 100,
and anything at or above the **threshold** goes to the alert queue.

| Rule | What it looks for | Why it matters |
|---|---|---|
| Unusual amount | Amount far above *this customer's* typical spend for that transaction type (robust z-score on a log scale: partial points from z = {scoring.Z_FIRE:g}, full at z = {scoring.Z_FULL:g}) | Account takeover often starts with a big purchase or transfer |
| Velocity | {scoring.VELOCITY_MIN_COUNT}+ transactions within a 1-hour window | Fraudsters use a stolen card fast, before it's blocked |
| Late night | 12:00-4:59 AM | Unusual for most customers; common for card-testing and ATM cash-outs |
| New location / merchant | First time in a city and/or at a merchant (after a customer's first {scoring.NEW_PLACE_WARMUP} transactions) | Card cloning and online fraud show up in unfamiliar places |
| Structuring | Cash deposit of $9,000-$9,999.99 | Avoiding the $10,000 Currency Transaction Report is a federal crime (31 U.S.C. 5324) |

Customers with fewer than {scoring.MIN_HISTORY} transactions skip the amount and new-place
checks, because we don't yet know what "normal" looks like for them.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("🔎 Transaction Monitoring & Fraud Risk Scoring")
    st.caption("FIN451 FinTech prototype. Rule-based risk scores with plain-English explanations.")

    uploaded_bytes, weights, threshold = sidebar_controls()

    try:
        signals, warnings = prepare_data(uploaded_bytes,
                                         None if uploaded_bytes else data_file_version())
    except scoring.DataValidationError as e:
        st.error(f"**We couldn't use this file.** {e}")
        show_expected_format()
        st.stop()
    except Exception as e:   # last-resort safety net so the app never shows a raw crash
        st.error(f"**Something unexpected went wrong while reading the data:** {e}")
        show_expected_format()
        st.stop()

    if warnings:
        with st.expander(f"⚠️ Data quality notes ({len(warnings)})", expanded=uploaded_bytes is not None):
            for w in warnings:
                st.write(f"- {w}")

    scored = scoring.score_transactions(signals, weights, threshold)

    show_summary(scored)
    show_rules_explainer()
    st.divider()
    show_performance(scored)
    st.divider()
    show_charts(scored, threshold)
    st.divider()
    show_alert_queue(scored)


main()
