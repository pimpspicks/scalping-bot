"""Fourth set: ideas from crypto papers (second research round), at their published settings.

- Intraday time-series momentum (Shen, Urquhart & Wang 2022): the first half hour of the day predicts the
  last half hour. Here the UTC day: at 23:30 trade in the direction of 00:00-00:30.
- Overnight session (Quantpedia): BTC's gains come outside 10:00-16:00 New York on stock-market days.
  Long at 16:00 New York on weekdays.
- Abnormal-day momentum (Caporale & Plastun 2020): on a day whose move so far is beyond 2 standard deviations
  of daily returns (last 90 days), trade with it, once per day.
- Distance to the 1-week high (Ficura 2023): for large coins, closeness to the 1-week high predicts gains.
  At the daily close, long on a new 7-day closing high, short on a new 7-day closing low.
- MAX momentum (Li et al.): coins with the biggest recent daily jump keep outperforming. Each Monday 00:00 UTC,
  long the coin whose largest daily return over the last 30 days is the highest of BTC/ETH/SOL, short the lowest.

The quarter-hour effect (Shanaev et al. 2023; a 2026 arXiv paper on futures) is a 1-minute pattern that is
measured in `quarter_hour_study`, since its holding time is far too short for the ladder.
"""
import numpy as np
import pandas as pd

from .rules2 import _per_hour, _sig
from .rules3 import _closes_at


def _daily_close(df):
    """UTC daily closes, indexed by the day."""
    return df.close.resample("1D").last()


def paper_rule_signals(symbol: str, all_bars: dict) -> dict[tuple[str, str], np.ndarray]:
    b = all_bars[symbol]
    df = b.df
    c = df.close
    ph = _per_hour(df)
    n = len(df)
    R = {}
    if ph < 2:  # needs bars of 30 minutes or less
        return R
    ct = (df.index + (df.index[1] - df.index[0])).tz_convert("UTC")
    day = ct.floor("D")

    # Intraday momentum: sign of 00:00-00:30 UTC, traded at 23:30 UTC.
    at0030 = _closes_at_min(ct, 0, 30)
    at2330 = _closes_at_min(ct, 23, 30)
    first = c / pd.Series(np.where(_closes_at(df, "UTC", None, 0), c, np.nan), index=df.index).ffill() - 1
    first_ret = pd.Series(np.where(at0030, first, np.nan), index=df.index).ffill()
    R[("intraday momentum", "First half hour -> last half hour")] = _sig(at2330 & (first_ret > 0), at2330 & (first_ret < 0))

    # Overnight session: long at the US stock close on weekdays.
    us_close = _closes_at(df, "America/New_York", None, 16) & np.asarray(
        (ct.tz_convert("America/New_York").dayofweek < 5))
    R[("overnight session", "Long at the US stock close")] = _sig(us_close, np.zeros(n, bool))

    # Abnormal day: move since 00:00 UTC beyond 2 sd of daily returns over the last 90 days.
    d_close = _daily_close(df)
    d_ret = np.log(d_close).diff()
    sd = d_ret.rolling(90, min_periods=30).std().shift()  # known at the start of each day
    sd_bar = pd.Series(sd.reindex(day).to_numpy(), index=df.index)
    open_px = pd.Series(np.where(_closes_at(df, "UTC", None, 0), c, np.nan), index=df.index).ffill()
    move = np.log(c / open_px)
    big_up, big_dn = move > 2 * sd_bar, move < -2 * sd_bar
    first_up = big_up & (big_up.astype(int).groupby(day).cumsum() == 1)
    first_dn = big_dn & (big_dn.astype(int).groupby(day).cumsum() == 1)
    R[("abnormal day", "Follow a 2-sd day, once")] = _sig(first_up, first_dn)

    # 1-week high / low at the UTC daily close.
    midnight = _closes_at(df, "UTC", None, 0)
    hi7 = d_close.rolling(7).max().shift()
    lo7 = d_close.rolling(7).min().shift()
    # the daily close known at 00:00 is the close of the day that just ended
    prev_day = (ct - pd.Timedelta("1min")).floor("D")
    dc = pd.Series(d_close.reindex(prev_day).to_numpy(), index=df.index)
    h7 = pd.Series(hi7.reindex(prev_day).to_numpy(), index=df.index)
    l7 = pd.Series(lo7.reindex(prev_day).to_numpy(), index=df.index)
    R[("1-week high", "New 7-day closing high / low")] = _sig(midnight & (dc > h7), midnight & (dc < l7))

    # MAX momentum across the three coins, weekly.
    monday = _closes_at(df, "UTC", 0, 0)
    mx = {}
    for s, bb in all_bars.items():
        dr = _daily_close(bb.df).pct_change()
        mx[s] = dr.rolling(30).max()
    mx = pd.DataFrame(mx)
    rank = mx.rank(axis=1)
    top, bottom = rank[symbol] == len(all_bars), rank[symbol] == 1
    t_bar = pd.Series(top.reindex(prev_day).to_numpy(), index=df.index).fillna(False).astype(bool)
    b_bar = pd.Series(bottom.reindex(prev_day).to_numpy(), index=df.index).fillna(False).astype(bool)
    R[("max momentum", "Long highest 30-day MAX coin, short lowest")] = _sig(monday & t_bar, monday & b_bar)
    return {k: np.nan_to_num(np.asarray(s, float)).astype(int) for k, s in R.items()}


def _closes_at_min(ct, hour, minute) -> np.ndarray:
    return np.asarray((ct.hour == hour) & (ct.minute == minute))


def quarter_hour_study(minutes: pd.DataFrame) -> pd.DataFrame:
    """Average 1-minute return (bps) by minute of the hour, per year."""
    r = np.log(minutes.close).diff() * 1e4
    g = pd.DataFrame({"r": r, "minute": minutes.index.minute, "year": minutes.index.year}).dropna()
    g["quarter_turn"] = g.minute % 15 == 0
    out = g.groupby(["year", "quarter_turn"]).r.agg(["mean", "std", "count"])
    out["t_stat"] = out["mean"] / (out["std"] / np.sqrt(out["count"]))
    return out
