"""Local HTTP entry point for the PromoAI dashboard and Python solver."""
import csv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Semaphore
import webbrowser

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 8_000_000
RUN = Semaphore(1)
FIELDS = ("caps", "pepsi-cap", "weekly-cap", "budget", "pepsi-floor", "retailer-floor",
          "share-floor", "min-duration", "max-duration", "spacing", "weights")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def solve(inputs):
    custom = bool(inputs.get("history"))
    if custom != bool(inputs.get("offers")):
        raise ValueError("Upload history and eligible offers together.")
    if not custom and inputs.get("locks"):
        raise ValueError("Upload both custom CSVs when supplying custom locks.")
    with TemporaryDirectory(prefix="promoai-") as directory:
        work = Path(directory)
        source = work if custom else ROOT / "data"
        if custom:
            for key in ("history", "offers"):
                (source / (key + ".csv")).write_text(inputs[key], encoding="utf-8")
        offers = read_csv(source / "offers.csv")
        if not offers or len(offers) > 2500:
            raise ValueError("Supply between 1 and 2,500 eligible offer rows.")
        locks_file = source / "locks.json"
        if custom:
            locks_file.write_text(inputs.get("locks") or "[]", encoding="utf-8")
        locks = json.loads(locks_file.read_text())
        if not isinstance(locks, list) or any(not isinstance(lock, list) or len(lock) != 3 for lock in locks):
            raise ValueError("Locks must be a JSON list of [group, week, offer] rows.")
        groups = {row["group"]: {"group": row["group"], "owned": row["owned"]} for row in offers}
        dates = sorted({row["week_start"] for row in offers})
        week_by_date = {date: index for index, date in enumerate(dates, 1)}
        locked = {(g, w): offer for g, w, offer in locks}
        if any(row["owned"] == "0" and row["offer"] != "Base" and
               (row["group"], week_by_date[row["week_start"]]) not in locked for row in offers):
            raise ValueError("Lock a competitor offer for every week where it has an eligible discount.")
        output = work / "results"
        cmd = [sys.executable, str(ROOT / "PromoAI_Model.py"), "--history", str(source / "history.csv"),
               "--offers", str(source / "offers.csv"), "--locks", str(locks_file),
               "--output-dir", str(output)]
        for key in FIELDS:
            values = inputs[key].split()
            if not values or not all(re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", v) for v in values):
                raise ValueError("Enter numeric values for " + key + ".")
            if (key == "caps" and len(values) > 3) or (key not in ("caps", "weights") and len(values) != 1):
                raise ValueError("Check the number of values for " + key + ".")
            cmd.extend(["--" + key, *values])
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=210)
        if process.returncode:
            error = process.stderr.strip().splitlines()
            raise ValueError(error[-1] if error else "The solver could not complete this run.")
        scenarios = {}
        for row in read_csv(output / "scenario_status.csv"):
            cap = row["promotion_cap"]
            scenario = {"status": row["status"]}
            if row["solved"] == "True":
                scenario["calendar"] = read_csv(output / f"calendar_cap_{cap}.csv")
                scenario["metrics"] = {r[""]: float(r["value"]) for r in read_csv(output / f"metrics_cap_{cap}.csv")}
            scenarios[cap] = scenario
        return {"catalog": list(groups.values()), "offers": offers, "locks": locks,
                "scenarios": scenarios, "summary": "\n".join(process.stdout.splitlines()[:2])}


class Handler(SimpleHTTPRequestHandler):
    def json_response(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            self.json_response({"live": True})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/solve":
            return self.json_response({"error": "Unknown endpoint."}, 404)
        if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            return self.json_response({"error": "Use the local dashboard to run the solver."}, 403)
        if not RUN.acquire(blocking=False):
            return self.json_response({"error": "A scenario run is already in progress."}, 409)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BYTES:
                raise ValueError("Upload limit is 8 MB per run.")
            self.json_response(solve(json.loads(self.rfile.read(size))))
        except (ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            self.json_response({"error": str(error)}, 400)
        finally:
            RUN.release()


if __name__ == "__main__":
    address = ("127.0.0.1", 8765)
    with ThreadingHTTPServer(address, partial(Handler, directory=ROOT / "dist")) as server:
        url = f"http://{address[0]}:{address[1]}"
        print(f"PromoAI planner at {url}", flush=True)
        webbrowser.open(url)
        server.serve_forever()
