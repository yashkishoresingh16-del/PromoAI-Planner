# PromoAI Promotion Planner

The hosted dashboard shows verified sample calendars for the synthetic supermarket.
For live CSV uploads and new optimization runs, use the Python version locally.
The local dashboard sends uploads only to the loopback server and removes them
after solving. Use INR and the same units throughout.

On Windows, extract the ZIP and run from Command Prompt:

    py -m pip install -r requirements.txt
    py app.py

The browser opens at http://127.0.0.1:8765. Expand **Run OR-Tools on your business inputs**.
Leave files blank to rerun the included sample. Use the three files in data as schemas.
History includes one record per group and observed week, with units, offer,
shelf price, own discount, peer discount, segment, and holiday flag. Future offers
include each eligible group-week-option and per-unit revenue, margin, and funding.
Every group-week requires a Base option. Use matching labels between the two files.
Locks fix competitor decisions where eligible discounts exist. Use assumptions
explicitly when margins, costs, or competitor plans are unknown.

The prototype follows the PromoAI sequence: weekly LightGBM forecast, piecewise
linear peer-discount response, and OR-Tools MPSolver with SCIP. The sample uses a
fictional retailer and 12 weeks. Nine competitor slots are committed. Cap 25 and
cap 16 are feasible; cap 8 is infeasible. Holdout accuracy describes simulated
sales only. A new solve allows up to 60 seconds per scenario.
