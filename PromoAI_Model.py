"""PromoAI classroom replication. Install with: pip install ortools lightgbm numpy pandas scikit-learn
Run: python PromoAI_Model.py
"""
import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from ortools.linear_solver import pywraplp

GROUPS = {
    "Cola": ("beverages", True, 100),
    "Rival Cola": ("beverages", False, 82),
    "Chips": ("snacks", True, 92),
    "Rival Chips": ("snacks", False, 76),
}
OFFERS = {"Base": 0., "10% Off": .10, "20% Off": .20}
FEATURES = ["group", "segment", "offer", "week_of_year", "holiday",
            "discount", "peer_discount", "shelf_price"]
WEEKS = list(range(1, 7))
PRESSURE = np.array([0., .05, .10, .15, .20])


def peers(group):
    return [other for other in GROUPS
            if other != group and GROUPS[other][0] == GROUPS[group][0]]


def features(group, week, offer, pressure):
    segment, _, _ = GROUPS[group]
    discount = OFFERS[offer]
    seasonal_week = (week - 1) % 52 + 1
    return dict(group=group, segment=segment, offer=offer,
                week_of_year=seasonal_week, holiday=int(seasonal_week in (4, 25, 51)),
                discount=discount, peer_discount=pressure, shelf_price=50 * (1 - discount))


def feature_frame(rows):
    frame = pd.DataFrame(rows)[FEATURES].copy()
    for name, categories in {
        "group": list(GROUPS), "segment": sorted({item[0] for item in GROUPS.values()}), "offer": list(OFFERS)
    }.items():
        frame[name] = pd.Categorical(frame[name], categories=categories)
    return frame


def train_forecast(history=None):
    """Simulate weekly historical records, then hold out the most recent weeks."""
    if history is None:
        rng = np.random.default_rng(23)
        records = []
        for week in range(1, 261):
            offers = dict(zip(GROUPS, rng.choice(list(OFFERS), len(GROUPS), p=[.5, .3, .2])))
            for group, offer in offers.items():
                pressure = np.mean([OFFERS[offers[other]] for other in peers(group)])
                row = features(group, week, offer, pressure)
                baseline = GROUPS[group][2]
                demand = baseline * (1 + 4.2 * row["discount"] - 1.9 * pressure
                                      + .26 * row["holiday"]
                                      + .10 * np.sin(2 * np.pi * row["week_of_year"] / 52))
                records.append(dict(row, period=week, units=max(1., demand + rng.normal(0, 5))))
        history = pd.DataFrame(records)
    periods = sorted(history.period.unique())
    if len(periods) < 12:
        raise ValueError("Provide at least 12 distinct historical weeks.")
    cutoff = periods[-max(2, len(periods) // 8)]
    train = history[history.period < cutoff]
    valid = history[history.period >= cutoff]
    model = LGBMRegressor(objective="mape", n_estimators=160, learning_rate=.05,
                          num_leaves=15, min_child_samples=12,
                          verbosity=-1, random_state=23, n_jobs=1)
    model.fit(feature_frame(train), train.units)
    prediction = model.predict(feature_frame(valid))
    positive = valid.units > 0
    mape = float(np.mean(abs(prediction[positive] - valid.units[positive]) / valid.units[positive])) if positive.any() else np.nan
    denominator = valid.units.sum()
    wape = float(abs(prediction - valid.units).sum() / denominator) if denominator else np.nan
    baseline = valid.group.map(train.groupby("group").units.median())
    report = valid[["week_start", "group", "offer", "units"]].copy() if "week_start" in valid else valid[["group","offer","units"]].copy()
    report["forecast_units"] = prediction
    report["baseline_units"] = baseline
    baseline_wape = float(abs(baseline - valid.units).sum() / denominator) if denominator else np.nan
    return model, mape, wape, baseline_wape, report


def offer_curves(model, keys, planning_features=None):
    rows = [features(group, week, offer, pressure) if planning_features is None
            else dict(planning_features[group, week, offer], group=group, offer=offer,
                      peer_discount=pressure)
            for group, week, offer in keys for pressure in PRESSURE]
    forecasts = np.maximum(0, model.predict(feature_frame(rows)))
    return dict(zip(keys, forecasts.reshape(len(keys), len(PRESSURE))))


def unit_financials(group, offer):
    """PepsiCo sales/margin, retailer sales/margin, manufacturer funding per unit."""
    discount = OFFERS[offer]
    shelf = 50 * (1 - discount)
    sell_in = 32 - 20 * discount
    owned = GROUPS[group][1]
    return np.array([sell_in if owned else 0., sell_in - 22 if owned else 0.,
                     shelf, shelf - sell_in, 32 - sell_in if owned else 0.])


def optimize(model, cap, pepsi_cap, pepsi_floor, retailer_floor, share_floor,
             weekly_cap, min_duration, max_duration, spacing, budget, weights,
             planning=None, locks=None):
    keys = list(product(GROUPS, WEEKS, OFFERS)) if planning is None else list(planning)
    curves = offer_curves(model, keys, planning)
    upper, integral, objective, rows, lower_rows, upper_rows = [], [], [], [], [], []

    def variable(binary=False, coefficient=0.):
        index = len(upper)
        upper.append(1.)
        integral.append(int(binary))
        objective.append(coefficient)
        return index

    def constrain(coefficients, low=-np.inf, high=np.inf):
        row = {}
        for index, coefficient in coefficients:
            row[index] = row.get(index, 0.) + coefficient
        rows.append(row)
        lower_rows.append(low)
        upper_rows.append(high)

    def exact(terms, value):
        constrain(terms, value, value)

    def totals(indices, factor=1.):
        return [(index, factor) for index in indices]

    x = {key: variable(binary=True) for key in keys}
    pressure_terms = {(g, w): [] for g, w in product(GROUPS, WEEKS)}
    finance = []
    for group, week, offer in keys:
        key = group, week, offer
        values = unit_financials(group, offer) if planning is None else np.array([
            planning[key][field] for field in FINANCIAL_FIELDS
        ])
        segments = []
        for k in range(len(PRESSURE) - 1):
            segment = variable(binary=True)
            endpoints = []
            for j in (k, k + 1):
                demand = curves[key][j]
                contribution = values * demand
                weight = variable(coefficient=-float(weights @ contribution[:4]))
                endpoints.append(weight)
                pressure_terms[group, week].append((weight, PRESSURE[j]))
                finance.append((key, weight, demand, contribution))
            # Exactly one interval for a selected offer; interpolation uses its endpoints.
            exact(totals(endpoints) + [(segment, -1.)], 0.)
            segments.append(segment)
        exact(totals(segments) + [(x[key], -1.)], 0.)

    options = {(g, w): [] for g, w in product(GROUPS, WEEKS)}
    for key in keys:
        options[key[:2]].append(key)
    active = {group_week: [x[key] for key in choices if key[2] != "Base"]
              for group_week, choices in options.items()}
    for group, week in active:
        exact(totals([x[key] for key in options[group, week]]), 1.)
        related = peers(group)
        peer_terms = [
            (x[key], -(OFFERS[key[2]] if planning is None else planning[key]["discount"]) / len(related))
            for other in related for key in options[other, week]
        ]
        exact(pressure_terms[group, week] + peer_terms, 0.)

    constrain(totals([index for indices in active.values() for index in indices]), high=cap)
    constrain(totals([index for (g, _), indices in active.items() if GROUPS[g][1]
                      for index in indices]), high=pepsi_cap)
    for week in WEEKS:
        constrain(totals([index for group in GROUPS for index in active[group, week]]),
                  high=weekly_cap)

    # The benchmark fixes one competitor offer. Base assignments can also be locked.
    for key in ([("Rival Cola", 4, "10% Off")] if locks is None else locks):
        exact([(x[key], 1.)], 1.)
    constrain([(i, f[1] - pepsi_floor * f[0]) for _, i, _, f in finance], low=0.)
    constrain([(i, f[3] - retailer_floor * f[2]) for _, i, _, f in finance], low=0.)
    constrain([(i, f[2] * (int(GROUPS[key[0]][1]) - share_floor))
               for key, i, _, f in finance], low=0.)
    constrain([(i, f[4]) for _, i, _, f in finance], high=budget)

    for group in GROUPS:
        if not GROUPS[group][1]:
            continue
        for week in WEEKS:
            current = active[group, week]
            previous = active.get((group, week - 1), [])
            start = variable(binary=True)
            constrain(totals(current) + totals(previous, -1.) + [(start, -1.)], high=0.)
            constrain([(start, 1.)] + totals(current, -1.), high=0.)
            constrain([(start, 1.)] + totals(previous), high=1.)
            for future in range(week, week + min_duration):
                constrain([(start, 1.)] + totals(active.get((group, future), []), -1.), high=0.)
            if week + max_duration <= WEEKS[-1]:
                window = [i for w in range(week, week + max_duration + 1) for i in active[group, w]]
                constrain(totals(window), high=max_duration)
            # Detect the first base week after an active block and enforce the full gap.
            for future in range(week + 1, min(week + spacing, WEEKS[-1] + 1)):
                constrain(totals(previous) + totals(current, -1.) + totals(active[group, future]), high=1.)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP backend unavailable. Install the standard OR-Tools wheel.")
    solver.SetTimeLimit(60_000)
    solver.SetNumThreads(1)
    variables = [solver.BoolVar(f"x_{i}") if binary else solver.NumVar(0., upper[i], f"v_{i}")
                 for i, binary in enumerate(integral)]
    for row, low, high in zip(rows, lower_rows, upper_rows):
        constraint = solver.RowConstraint(float(low), float(high), "")
        for i, coefficient in row.items():
            constraint.SetCoefficient(variables[i], float(coefficient))
    goal = solver.Objective()
    for i, coefficient in enumerate(objective):
        goal.SetCoefficient(variables[i], -coefficient)
    goal.SetMaximization()
    status = solver.Solve()
    labels = {solver.OPTIMAL: "OPTIMAL", solver.FEASIBLE: "FEASIBLE (not proven optimal)",
              solver.INFEASIBLE: "INFEASIBLE", solver.NOT_SOLVED: "NOT SOLVED",
              solver.ABNORMAL: "ABNORMAL", solver.UNBOUNDED: "UNBOUNDED"}
    message = f"OR-Tools SCIP: {labels.get(status, status)}"
    if status not in (solver.OPTIMAL, solver.FEASIBLE):
        return message, None, None
    if not solver.VerifySolution(1e-6, True):
        raise RuntimeError("SCIP returned a solution that failed validation.")
    solution = [variable.solution_value() for variable in variables]
    chosen = {key for key, index in x.items() if solution[index] > .5}
    forecast = {key: 0. for key in chosen}
    total = np.zeros(5)
    owned_retail_revenue = 0.
    for key, index, demand, values in finance:
        total += solution[index] * values
        if GROUPS[key[0]][1]:
            owned_retail_revenue += solution[index] * values[2]
        if key in chosen:
            forecast[key] += solution[index] * demand
    calendar = pd.DataFrame([
        {"group": g, "week": w, "offer": o, "forecast_units": forecast[g, w, o]}
        for g, w, o in sorted(chosen)
    ])
    metrics = dict(zip(["PepsiCo revenue", "PepsiCo margin", "Retailer revenue",
                        "Retailer margin", "Trade spend"], total))
    metrics["Weighted score"] = weights @ total[:4]
    metrics["PepsiCo share of retail revenue"] = owned_retail_revenue / total[2] if total[2] else np.nan
    metrics["Active promotion weeks"] = sum(offer != "Base" for _, _, offer in chosen)
    metrics["Forecast volume"] = calendar.forecast_units.sum()
    return message, calendar, metrics


FINANCIAL_FIELDS = ["pepsi_unit_revenue", "pepsi_unit_margin", "retail_unit_revenue",
                    "retail_unit_margin", "trade_spend_per_unit"]


def read_business_inputs(history_path, planning_path):
    """Read one retailer's historical observations and eligible future offers."""
    global GROUPS, OFFERS, WEEKS, PRESSURE
    history = pd.read_csv(history_path)
    planning = pd.read_csv(planning_path)
    for frame, required, label in (
        (history, {"week_start", "units", *FEATURES} - {"week_of_year"}, "history"),
        (planning, {"week_start", "owned", *FINANCIAL_FIELDS, *FEATURES}
         - {"week_of_year", "peer_discount"}, "offers"),
    ):
        missing = required.difference(frame.columns)
        if frame.empty or missing or frame[list(required)].isna().any().any():
            raise ValueError(f"{label}: provide nonempty required columns. Missing: {sorted(missing)}")
        frame["week_start"] = pd.to_datetime(frame.week_start, format="%Y-%m-%d")
        frame["week_of_year"] = frame.week_start.dt.isocalendar().week.astype(int)
        numeric = ["discount", "shelf_price", "holiday"] + (
            ["units", "peer_discount"] if label == "history" else ["owned", *FINANCIAL_FIELDS])
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: numeric inputs must be finite.")
        if not frame.discount.between(0, 1).all() or (frame.shelf_price <= 0).any():
            raise ValueError(f"{label}: use discounts from 0 to 1 and positive shelf prices.")
        if not frame.holiday.isin([0, 1]).all():
            raise ValueError(f"{label}: holiday must be 0 or 1.")
    if (history.units < 0).any() or not history.peer_discount.between(0, 1).all():
        raise ValueError("Historical units must be nonnegative and peer discount in [0, 1].")
    if history.duplicated(["group", "week_start"]).any():
        raise ValueError("History requires one observed offer per group and week.")
    if not planning.owned.isin([0, 1]).all():
        raise ValueError("Future offers require owned to be 0 or 1.")
    if (planning[["pepsi_unit_revenue", "retail_unit_revenue", "trade_spend_per_unit"]] < 0).any().any():
        raise ValueError("Revenue and trade spend per unit must be nonnegative.")
    if (planning.loc[planning.owned == 0, FINANCIAL_FIELDS[:2] + [FINANCIAL_FIELDS[-1]]] != 0).any().any():
        raise ValueError("PepsiCo revenue, margin and funding must be zero for competitor products.")
    if (planning.groupby("group")[["segment", "owned"]].nunique() != 1).any().any():
        raise ValueError("Segment and ownership must be consistent for a group.")
    dates = sorted(planning.week_start.unique())
    if min(dates) <= history.week_start.max() or any(
        (right - left) != np.timedelta64(7, "D") for left, right in zip(dates, dates[1:])
    ):
        raise ValueError("Future weeks must be consecutive and follow the historical data.")
    if planning.duplicated(["group", "week_start", "offer"]).any():
        raise ValueError("Future offers require unique group-week-offer rows.")
    if (planning.loc[planning.offer == "Base", "discount"] != 0).any():
        raise ValueError("Base must have zero discount.")
    GROUPS = {g: (row.segment, bool(row.owned), 0)
              for g, row in planning.drop_duplicates("group").set_index("group").iterrows()}
    OFFERS = dict.fromkeys(pd.concat([history.offer, planning.offer]).unique())
    WEEKS = list(range(1, len(dates) + 1))
    calendar_dates = dict(zip(WEEKS, dates))
    planning["week"] = planning.week_start.map({date: week for week, date in calendar_dates.items()})
    eligible = planning.set_index(["group", "week", "offer"]).to_dict("index")
    if any((g, w, "Base") not in eligible for g, w in product(GROUPS, WEEKS)):
        raise ValueError("Each group and future week needs an eligible Base row.")
    history = history[history.group.isin(GROUPS)].copy()
    if set(GROUPS) - set(history.group):
        raise ValueError("Each future product group needs historical observations.")
    expected_segments = history.group.map({g: value[0] for g, value in GROUPS.items()})
    if not history.segment.eq(expected_segments).all():
        raise ValueError("Product segment must match between history and future offers.")
    history["period"] = history.week_start
    cutoff = sorted(history.period.unique())[-max(2, len(history.period.unique()) // 8)] if history.period.nunique() >= 12 else None
    if cutoff is not None:
        train_rows = history[history.period < cutoff]
        seen = set(zip(train_rows.group, train_rows.offer))
        missing_support = {(g, o) for g, _, o in eligible} - seen
        if missing_support:
            raise ValueError(f"Future group/offer combinations absent from training: {sorted(missing_support)}")
    PRESSURE = np.linspace(0., max(.000001, float(planning.discount.max())), 5)
    if planning.discount.max() > history.peer_discount.max():
        print("Warning: some candidate peer discounts exceed observed historical pressure.")
    return history, eligible, calendar_dates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", nargs="+", type=int, default=[7, 4])
    parser.add_argument("--pepsi-cap", type=int, default=6)
    parser.add_argument("--weekly-cap", type=int, default=2)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--max-duration", type=int, default=3)
    parser.add_argument("--spacing", type=int, default=1)
    parser.add_argument("--pepsi-floor", type=float, default=.24)
    parser.add_argument("--retailer-floor", type=float, default=.29)
    parser.add_argument("--share-floor", type=float, default=.45)
    parser.add_argument("--budget", type=float, default=2500.)
    parser.add_argument("--weights", nargs=4, type=float, default=[.25] * 4,
                        metavar=("PEPSICO_REV", "PEPSICO_MARGIN", "RETAIL_REV", "RETAIL_MARGIN"))
    parser.add_argument("--history", type=Path, help="Historical weekly sales CSV")
    parser.add_argument("--offers", type=Path, help="Future eligible offers CSV")
    parser.add_argument("--locks", type=Path, help="JSON list of [group, week number, offer] locks")
    parser.add_argument("--output-dir", type=Path, help="Write a calendar and metrics CSV per feasible scenario")
    args = parser.parse_args()
    if bool(args.history) != bool(args.offers):
        parser.error("Provide --history and --offers together.")
    history = planning = dates = locks = None
    if args.history:
        try:
            history, planning, dates = read_business_inputs(args.history, args.offers)
            locks = [tuple(lock) for lock in json.loads(args.locks.read_text())] if args.locks else []
            if any(lock not in planning for lock in locks):
                raise ValueError("Each locked group-week-offer must exist in the future offers file.")
        except (ValueError, OSError, TypeError) as error:
            parser.error(str(error))
    elif args.locks:
        parser.error("--locks requires the real-data CSV inputs.")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(args.weights)
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(sum(weights), 1)):
        parser.error("The four financial weights must be nonnegative and sum to 1.")
    if not 1 <= args.min_duration <= args.max_duration or args.spacing < 0:
        parser.error("Require 1 <= min-duration <= max-duration and spacing >= 0.")
    if min(*args.caps, args.pepsi_cap, args.weekly_cap) < 0:
        parser.error("Promotion caps must be nonnegative.")
    if any(not 0 <= floor <= 1 for floor in (args.pepsi_floor, args.retailer_floor, args.share_floor)):
        parser.error("Margin and share floors must be between zero and one.")
    if not np.isfinite(args.budget) or args.budget < 0:
        parser.error("The trade-spend budget must be finite and nonnegative.")
    model, mape, wape, baseline_wape, forecast_test = train_forecast(history)
    label = "Supplied historical data" if history is not None else "Synthetic historical data"
    print(f"{label} | held-out MAPE on positive units: {mape:.1%} | WAPE: {wape:.1%}")
    print(f"Training-group-median baseline WAPE: {baseline_wape:.1%}")
    if args.output_dir:
        forecast_test.to_csv(args.output_dir / "forecast_holdout.csv", index=False)
    scenario_status = []
    for cap in args.caps:
        message, calendar, metrics = optimize(
            model, cap, args.pepsi_cap, args.pepsi_floor, args.retailer_floor,
            args.share_floor, args.weekly_cap, args.min_duration, args.max_duration,
            args.spacing, args.budget, weights, planning, locks,
        )
        print(f"\nPromotion cap {cap} | {message}")
        scenario_status.append({"promotion_cap": cap, "solved": calendar is not None, "status": message})
        if calendar is not None:
            if dates:
                calendar["week_start"] = calendar.week.map(dates)
            if args.output_dir:
                calendar.to_csv(args.output_dir / f"calendar_cap_{cap}.csv", index=False)
                pd.Series(metrics, name="value").to_csv(args.output_dir / f"metrics_cap_{cap}.csv")
            print(calendar.pivot(index="group", columns="week", values="offer").to_string())
            for label, amount in metrics.items():
                print(f"{label}: {amount:,.2f}")

    if args.output_dir:
        pd.DataFrame(scenario_status).to_csv(args.output_dir / "scenario_status.csv", index=False)


if __name__ == "__main__":
    main()
