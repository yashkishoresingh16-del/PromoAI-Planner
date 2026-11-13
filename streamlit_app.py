"""Streamlit interface for the PromoAI planning model."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import solve

ROOT = Path(__file__).resolve().parent
LABELS = {
    "Forecast volume": "Forecast units",
    "Trade spend": "Trade spend (₹)",
    "PepsiCo revenue": "PepsiCo revenue (₹)",
    "PepsiCo margin": "PepsiCo margin (₹)",
    "Retailer revenue": "Retailer revenue (₹)",
    "Retailer margin": "Retailer margin (₹)",
    "Weighted score": "Weighted score",
    "Active promotion weeks": "Active promotion slots",
    "PepsiCo share of retail revenue": "PepsiCo retail revenue share (%)",
}
st.set_page_config(page_title="PromoAI Promotion Planner", layout="wide")
st.title("PromoAI Promotion Planner")
st.write("Choose a promotion calendar by comparing forecast demand, financial results, and retailer constraints.")
sample = json.loads((ROOT / "dist" / "demo.json").read_text())
if "planner_result" not in st.session_state:
    st.session_state.planner_result = sample
    st.session_state.planner_source = "Synthetic demonstration · verified Google OR-Tools results"

with st.expander("Run a new Google OR-Tools optimization"):
    st.write("Leave both CSV uploads blank to use the included synthetic supermarket. For custom data, supply both CSVs and lock competitor discounts that the planner cannot control.")
    with st.form("planning"):
        c = st.columns(3)
        history = c[0].file_uploader("Weekly sales history", type="csv")
        offers_file = c[1].file_uploader("Future eligible offers", type="csv")
        locks = c[2].file_uploader("Competitor commitments", type="json")
        c = st.columns(4)
        caps = c[0].text_input("Total slot caps", "25 16 8", help="Enter up to three caps separated by spaces.")
        owner_cap = c[1].number_input("PepsiCo slot cap", 0, value=16)
        weekly = c[2].number_input("Weekly slot cap", 0, value=3)
        budget = c[3].number_input("Funding limit (₹)", 0, value=50000)
        c = st.columns(3)
        p_floor = c[0].number_input("PepsiCo margin floor (%)", 0, 100, 25)
        r_floor = c[1].number_input("Retailer margin floor (%)", 0, 100, 23)
        share = c[2].number_input("PepsiCo revenue share floor (%)", 0, 100, 45)
        c = st.columns(4)
        minimum = c[0].number_input("Minimum block weeks", 1, value=2)
        maximum = c[1].number_input("Maximum block weeks", 1, value=3)
        spacing = c[2].number_input("Base weeks between blocks", 0, value=1)
        weights = c[3].text_input("Financial weights", "0.25 0.25 0.25 0.25",
                                  help="PepsiCo revenue, PepsiCo margin, retailer revenue, retailer margin. Sum must equal 1.")
        run = st.form_submit_button("Train LightGBM and optimize", type="primary")
    if run:
        files = {"history": history, "offers": offers_file, "locks": locks}
        try:
            if sum(f.size for f in files.values() if f is not None) > 8_000_000:
                raise ValueError("The combined upload limit is 8 MB.")
            inputs = {key: file.getvalue().decode("utf-8-sig") if file else "" for key, file in files.items()}
            if any(file is not None and not inputs[key].strip() for key, file in files.items()):
                raise ValueError("Uploaded files must contain data.")
            inputs.update(caps=caps, budget=str(budget), weights=weights, **{
                "pepsi-cap": str(owner_cap), "weekly-cap": str(weekly),
                "pepsi-floor": str(p_floor/100), "retailer-floor": str(r_floor/100),
                "share-floor": str(share/100), "min-duration": str(minimum),
                "max-duration": str(maximum), "spacing": str(spacing),
            })
            with st.spinner("Training the forecast and solving the calendars…"):
                result = solve(inputs)
            st.session_state.planner_result = result
            st.session_state.planner_source = ("Custom business inputs" if history else "Synthetic demonstration") + " · live Google OR-Tools run"
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            st.error(str(error))
        except TimeoutError:
            st.error("The optimization exceeded the run limit. Try fewer weeks or scenarios.")
        except Exception as error:
            from subprocess import TimeoutExpired
            if not isinstance(error, TimeoutExpired):
                raise
            st.error("The optimization exceeded the run limit. Try fewer weeks or scenarios.")
    if st.button("Restore the sample scenarios"):
        st.session_state.planner_result = sample
        st.session_state.planner_source = "Synthetic demonstration · verified Google OR-Tools results"
        st.rerun()

data = st.session_state.planner_result
st.caption(st.session_state.planner_source)
if "summary" in data:
    st.caption(data["summary"])

with st.expander("Inspect planning inputs"):
    offer_frame = pd.DataFrame(data["offers"])
    group = st.selectbox("Product group", sorted(offer_frame.group.unique()))
    date = st.selectbox("Week starting", sorted(offer_frame.week_start.unique()))
    st.dataframe(offer_frame.loc[offer_frame.group.eq(group) & offer_frame.week_start.eq(date)], hide_index=True)
    st.markdown("**Competitor commitments**")
    st.dataframe(pd.DataFrame(data["locks"], columns=["Group", "Planning week", "Offer"]), hide_index=True)
    for name in ("history.csv", "offers.csv", "locks.json"):
        st.download_button("Download sample " + name, (ROOT / "data" / name).read_bytes(), name)

st.subheader("Recommended promotion calendar")
scenarios = data["scenarios"]
cap = st.selectbox("Total promotion slots", sorted(scenarios, key=int, reverse=True),
                   help="One slot means one product group on promotion for one week, including competitors.")
current = scenarios[cap]
st.write(current["status"])
if "calendar" in current:
    m = current["metrics"]
    c = st.columns(4)
    for column, key in zip(c, ("Forecast volume", "Trade spend", "Retailer margin", "Weighted score")):
        column.metric(LABELS[key], f"{m[key]:,.0f}")
    calendar = pd.DataFrame(current["calendar"])
    calendar["week"] = calendar["week"].astype(int)
    calendar["forecast_units"] = calendar["forecast_units"].astype(float)
    matrix = calendar.pivot(index="group", columns="week", values="offer").sort_index(axis=1)
    st.dataframe(matrix.style.map(lambda offer: "background-color: #def3e6; color: #12382b" if offer != "Base" else ""))
    st.download_button("Download selected calendar", calendar.to_csv(index=False), f"calendar_cap_{cap}.csv", "text/csv")
elif st.session_state.planner_source.endswith("verified Google OR-Tools results") and cap == "8":
    st.warning("Nine committed competitor slots already exceed the cap of eight.")

st.subheader("Compare feasible scenarios")
good = {f"Cap {cap}": scenario["metrics"] for cap, scenario in scenarios.items() if "metrics" in scenario}
if good:
    comparison = pd.DataFrame(good).reindex(LABELS)
    comparison.loc["PepsiCo share of retail revenue"] *= 100
    comparison = comparison.rename(index=LABELS)
    key = st.selectbox("Metric to visualize", list(LABELS.values()))
    st.bar_chart(comparison.loc[key].rename("Value").to_frame())
    st.dataframe(comparison.round(2))
else:
    st.info("Change the limits or commitments to create a feasible scenario.")
st.caption("Forecasts use LightGBM. The calendar is optimized by Google OR-Tools MPSolver with SCIP. Sample data is synthetic.")
