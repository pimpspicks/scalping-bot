"""US macro release calendar: FOMC statements (14:00 New York), CPI and jobs report (08:30 New York).

Dates are the actual release dates, read from the Fed's FOMC calendar pages and BLS's release archives (which
link each past release as cpi_MMDDYYYY.htm / empsit_MMDDYYYY.htm). Unscheduled Fed actions (March 2020
emergency cuts, facility announcements, the 2020 and 2025 strategy statements) are left out: a bot could not
have known about them in advance.
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

TZ = "America/New_York"
TIMES = {"FOMC": (14, 0), "CPI": (8, 30), "Jobs report": (8, 30)}
NOT_SCHEDULED = {"20200303", "20200315", "20200323", "20200331", "20200827", "20250822"}
FED = "https://www.federalreserve.gov/monetarypolicy/"
BLS = "https://www.bls.gov/bls/news-release/"
CSV = Path(__file__).with_name("econ_calendar.csv")


def _get(url: str) -> str:
    # BLS answers 403 to clients without a user agent naming the caller
    r = subprocess.run(["curl", "-sS", "-m", "60", "-A", "scalping-bot research", url], capture_output=True, text=True)
    return r.stdout


def fetch(first_year=2020) -> pd.DataFrame:
    rows = []
    pages = [_get(FED + "fomccalendars.htm")] + [_get(FED + f"fomchistorical{y}.htm") for y in range(first_year, 2021)]
    for d in sorted({m for p in pages for m in re.findall(r"monetary(\d{8})a\.htm", p)} - NOT_SCHEDULED):
        rows.append(("FOMC", d))
    for name, page in (("CPI", "cpi"), ("Jobs report", "empsit")):
        for m in sorted(set(re.findall(rf"{page}_(\d{{8}})\.htm", _get(BLS + f"{page}.htm")))):
            rows.append((name, m[4:] + m[:4]))
    df = pd.DataFrame(rows, columns=["event", "date"])
    df = df[df.date.str[:4].astype(int) >= first_year]
    return df.sort_values(["date", "event"]).reset_index(drop=True)


def load(path: Path = CSV) -> pd.DataFrame:
    """event, date (New York), time (UTC timestamp of the release)."""
    df = pd.read_csv(path, dtype=str)
    hm = df.event.map(TIMES)
    local = pd.to_datetime(df.date) + pd.to_timedelta([h * 60 + m for h, m in hm], unit="min")
    return df.assign(time=local.dt.tz_localize(TZ).dt.tz_convert("UTC"))
