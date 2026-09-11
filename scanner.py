#!/usr/bin/env python3
"""
Nifty 200 Swing Trade Scanner (cash market, no F&O)
====================================================
5-step funnel: market -> sector -> stock -> setup -> trade plan

Run:  python3 scanner.py            (after market close, Mon-Fri)
Deps: pandas, numpy, yfinance, requests
"""

import os, sys, json, time, datetime as dt
import numpy as np
import pandas as pd
import requests

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "output")
CONS_CSV   = os.path.join(BASE_DIR, "nifty200_constituents.csv")
CONS_URL   = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
EARN_CACHE = os.path.join(BASE_DIR, "earnings_cache.json")
POSITIONS  = os.path.join(BASE_DIR, "positions.csv")   # NEW: optional

# ----------------------------- tunable parameters -----------------------------
P = dict(
    history_days      = 400,
    min_turnover_cr   = 100,     # FIX: was 30 — thin names are risky
    atr_pct_min       = 2.0,
    atr_pct_max       = 6.0,
    max_dist_52w_high = 0.20,
    base_lookback     = 15,
    base_tightness    = 0.12,
    trig_proximity    = 0.04,
    breakout_vol_mult = 1.5,
    pb_window         = 40,
    pb_depth_min      = 0.04,
    pb_depth_max      = 0.12,
    pb_ma_tol         = 0.03,
    min_risk_pct      = 0.025,
    max_risk_pct      = 0.10,    # NEW: reject bases wider than 10%
    min_rr            = 2.0,
    min_rs_3m         = 5.0,     # NEW: must beat Nifty by >=5% over 3m
    earnings_blackout_days = 7,  # NEW: skip if results within N days

    # NEW: position sizing (edit for your account)
    capital           = 100_000, # INR
    risk_per_trade_pct = 0.5,    # % of capital risked per trade
)

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------- data loading --------------------------------
def load_constituents() -> pd.DataFrame:
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(CONS_URL, headers=hdr, timeout=20)
        r.raise_for_status()
        open(CONS_CSV, "wb").write(r.content)
    except Exception as e:
        if not os.path.exists(CONS_CSV):
            sys.exit(f"Cannot download Nifty 200 list and no cache found: {e}")
        print(f"[warn] using cached constituents ({e})")
    df = pd.read_csv(CONS_CSV).rename(columns=str.strip)
    df["YF"] = df["Symbol"].str.strip() + ".NS"
    return df[["Company Name", "Industry", "Symbol", "YF"]]

def download(tickers, period="400d") -> dict:
    import yfinance as yf
    out, failed = {}, []
    for i in range(0, len(tickers), 60):
        chunk = tickers[i:i+60]
        try:
            raw = yf.download(chunk, period=period, progress=False,
                              threads=True, auto_adjust=True)
        except Exception as e:
            print(f"[warn] chunk failed ({e}); retrying in 10s...")
            time.sleep(10)
            raw = yf.download(chunk, period=period, progress=False,
                              threads=False, auto_adjust=True)
        for t in chunk:
            try:
                d = raw.xs(t, level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"])
                if len(d) >= 220:
                    out[t] = d
                else:
                    failed.append(t)
            except Exception:
                failed.append(t)
    if failed:
        print(f"[info] skipped {len(failed)} tickers: "
              f"{', '.join(x.replace('.NS','') for x in failed[:8])}...")
    return out

# --------------------------- NEW: earnings cache -----------------------------
def load_earnings_cache() -> dict:
    if os.path.exists(EARN_CACHE):
        try:
            with open(EARN_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_earnings_cache(cache: dict):
    try:
        with open(EARN_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[warn] could not write earnings cache: {e}")

def next_earnings_date(yf_sym: str, cache: dict):
    """Return next earnings date as date, or None. Cache for 7 days."""
    import yfinance as yf
    today = dt.date.today()
    entry = cache.get(yf_sym)
    if entry and entry.get("checked"):
        try:
            checked = dt.date.fromisoformat(entry["checked"])
            if (today - checked).days < 7:
                nxt = entry.get("next")
                return dt.date.fromisoformat(nxt) if nxt else None
        except Exception:
            pass
    next_date = None
    try:
        cal = yf.Ticker(yf_sym).calendar
        if cal and isinstance(cal, dict) and "Earnings Date" in cal:
            ed = cal["Earnings Date"]
            if isinstance(ed, list) and ed:
                next_date = pd.to_datetime(ed[0]).date()
            elif ed is not None:
                next_date = pd.to_datetime(ed).date()
    except Exception:
        pass
    cache[yf_sym] = {
        "checked": today.isoformat(),
        "next": next_date.isoformat() if next_date else None,
    }
    return next_date

# ------------------------------- indicators ----------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["ATR"]   = tr.ewm(alpha=1/14, adjust=False).mean()
    df["ATRp"]  = df["ATR"] / c * 100
    df["V20"]   = v.rolling(20).mean()
    df["TURNcr"] = (c * v).rolling(20).mean() / 1e7
    df["RET63"] = c.pct_change(63) * 100
    df["RET21"] = c.pct_change(21) * 100
    df["H52"]   = h.rolling(252, min_periods=200).max()
    return df

# ------------------------------- setups ---------------------------------------
def analyze(sym, name, ind, df, nifty_ret63, top_sectors):
    """FIX: top_sectors passed in (was global). Returns dict or None."""
    df = df.dropna(subset=["SMA200", "ATR"])
    if len(df) < 100:
        return None
    x = df.iloc[-1]
    close, sma20, sma50, sma200 = x["Close"], x["SMA20"], x["SMA50"], x["SMA200"]
    atrp, turn, h52 = x["ATRp"], x["TURNcr"], x["H52"]
    ret63 = x["RET63"]
    rs63  = ret63 - nifty_ret63

    # ---- base filters ----
    if not (close > sma50 > sma200):
        return None
    if rs63 <= P["min_rs_3m"]:                                  # FIX: was rs63 <= 0
        return None
    if pd.isna(h52) or close < (1 - P["max_dist_52w_high"]) * h52:
        return None
    if not (P["atr_pct_min"] <= atrp <= P["atr_pct_max"]):
        return None
    if turn < P["min_turnover_cr"]:
        return None

    lb = P["base_lookback"]
    win = df.iloc[-lb-1:-1]                                     # FIX: exclude today
    base_hi, base_lo = win["High"].max(), win["Low"].min()
    base_height = base_hi - base_lo
    tight = (base_hi / base_lo - 1) <= P["base_tightness"]
    near_trig = close >= base_hi * (1 - P["trig_proximity"])

    v_ratio  = x["Volume"] / x["V20"] if x["V20"] else 0
    prior_hi = base_hi                                          # same as before, clearer
    fresh_break = (close > prior_hi) and v_ratio >= P["breakout_vol_mult"]
    vol_dryup = df["Volume"].iloc[-5:].mean() < x["V20"]

    hi60      = df["High"].iloc[-P["pb_window"]:].max()
    hi60_pos  = df["High"].iloc[-P["pb_window"]:].idxmax()
    depth     = hi60 / close - 1
    sw_lo     = df["Low"].iloc[-10:].min()

    # FIX: inline, no pm_ok()
    pb_ma_ok = ((abs(close/sma20 - 1) <= P["pb_ma_tol"] and sma20 > df["SMA20"].iloc[-6])
                or (abs(close/sma50 - 1) <= P["pb_ma_tol"] and sma50 > df["SMA50"].iloc[-6]))
    bounce = close > df["Close"].iloc[-2]

    setup, entry, stop, target = None, None, None, None
    fresh_break_flag = False

    # ---- Setup A: breakout ----
    if fresh_break:
        setup = "A: Fresh breakout"
        entry = round(close, 2)
        stop  = round(base_lo, 2)                               # FIX: was max(base_lo, entry*0.93)
        fresh_break_flag = True
    elif tight and near_trig:
        setup = "A: Base near trigger"
        entry = round(base_hi, 2)
        stop  = round(base_lo, 2)                               # FIX: was max(base_lo, entry*0.93)

    # ---- Setup B: pullback (real chart target) ----
    if setup is None:
        if (P["pb_depth_min"] <= depth <= P["pb_depth_max"]
                and pb_ma_ok and bounce and df.index[-1] > hi60_pos):
            e = round(df["High"].iloc[-1] * 1.002, 2)
            s = round(sw_lo * 0.995, 2)
            t = round(hi60, 2)
            if s < e and (e - s) / e >= P["min_risk_pct"]:
                setup, entry, stop, target = "B: Pullback to MA", e, s, t

    if setup is None or entry is None:
        return None

    # ---- risk checks ----
    risk_pct = (entry - stop) / entry * 100
    if risk_pct < P["min_risk_pct"] * 100:
        return None
    if risk_pct > P["max_risk_pct"] * 100:                      # NEW
        return None

    # ---- target ----
    if target is None:
        # FIX: base-scaled projection, not risk-scaled
        # near-trigger: entry=base_hi, stop=base_lo, so risk=base_height
        # target = entry + 2.5 * base_height gives R:R = 2.5 by design of the base
        target = round(entry + 2.5 * base_height, 2)

    rr = (target - entry) / (entry - stop)
    if rr < P["min_rr"]:
        return None

    # ---- composite score (0-100) ----
    score = 15 + min(max(rs63, 0), 30) / 30 * 30

    if setup.startswith("A"):
        if fresh_break_flag:
            score += 15
            score += 15 * min(v_ratio / 2.0, 1.0)
        else:
            tightness = 1 - min((base_hi/base_lo - 1) / P["base_tightness"], 1.0)
            score += tightness * 25
            score += 10 if vol_dryup else 0
            score += 10 * min(v_ratio / 1.5, 1.0)
    else:
        score += 15 * max(0, 1 - abs(depth - 0.06) / 0.06)
        score += 10 if pb_ma_ok else 0

    score += min(max(rr - 2, 0), 1) * 10
    in_top_sector = ind in top_sectors
    score += 10 if in_top_sector else 0

    return dict(Symbol=sym, Company=name, Industry=ind, Setup=setup,
                Close=round(close, 2), Entry=entry, Stop=stop,
                Target=target, Risk_pct=round(risk_pct, 1),
                RR=round(rr, 2), RS_3m=round(rs63, 1), ATR_pct=round(atrp, 1),
                Vol_x=round(v_ratio, 1), Turnover_cr=round(turn, 0),
                TopSector="Yes" if in_top_sector else "-",
                Score=round(min(score, 100), 0))

# --------------------------- NEW: positions check ----------------------------
def check_positions(frames) -> list:
    """Read positions.csv if present, return list of alert strings."""
    if not os.path.exists(POSITIONS):
        return []
    alerts = []
    try:
        pos = pd.read_csv(POSITIONS)
    except Exception as e:
        return [f"[warn] positions.csv unreadable: {e}"]
    for _, p in pos.iterrows():
        yf_sym = str(p["Symbol"]).strip() + ".NS"
        if yf_sym not in frames:
            continue
        close = frames[yf_sym].iloc[-1]["Close"]
        if close <= float(p["Stop"]):
            alerts.append(f"STOP HIT:  {p['Symbol']} close {close:,.2f} <= stop {p['Stop']}")
        elif close >= float(p["Target"]):
            alerts.append(f"TARGET HIT: {p['Symbol']} close {close:,.2f} >= target {p['Target']}")
    return alerts

# ------------------------------- main -----------------------------------------
def main():
    cons = load_constituents()
    tickers = cons["YF"].tolist()
    cache = os.path.join(BASE_DIR, f"data_cache_{dt.date.today().isoformat()}.pkl")
    if os.path.exists(cache):
        print(f"Loaded {len(cons)} Nifty 200 constituents. Using today's cached data.")
        data = pd.read_pickle(cache)
    else:
        print(f"Loaded {len(cons)} Nifty 200 constituents. Downloading data...")
        data = download(tickers + ["^NSEI"])
        pd.to_pickle(data, cache)
    nif = add_indicators(data.pop("^NSEI").copy())

    n_last = nif.iloc[-1]
    above50 = 0
    frames = {}
    for t, d in data.items():
        d = add_indicators(d.copy())
        frames[t] = d
        last = d.iloc[-1]
        if not pd.isna(last["SMA50"]) and last["Close"] > last["SMA50"]:
            above50 += 1
    breadth = above50 / len(frames) * 100

    mkt_green = (n_last["Close"] > n_last["SMA50"]) and (n_last["SMA20"] > n_last["SMA50"])
    mkt_note = ("GREEN-LIGHT: Nifty above 50-DMA, 20-DMA > 50-DMA"
                if mkt_green else
                "CAUTION: Nifty trend filter NOT green - cut size / skip new longs")

    # sector strength
    ind_ret = {}
    for t, d in frames.items():
        r = d.iloc[-1]["RET63"]
        if not pd.isna(r):
            ind_ret.setdefault(cons.loc[cons["YF"] == t, "Industry"].iloc[0], []).append(r)
    ind_table = (pd.DataFrame([(k, np.median(v), len(v)) for k, v in ind_ret.items()],
                              columns=["Industry", "MedianRet63", "N"])
                   .sort_values("MedianRet63", ascending=False))
    top_sectors = set(ind_table.head(5)["Industry"])            # FIX: no global

    nifty_ret63 = n_last["RET63"]
    rows = []
    for _, r in cons.iterrows():
        if r["YF"] not in frames:
            continue
        try:
            res = analyze(r["Symbol"], r["Company Name"], r["Industry"],
                          frames[r["YF"]], nifty_ret63, top_sectors)
            if res:
                rows.append(res)
        except Exception as e:
            print(f"[warn] {r['Symbol']}: {e}")

    out = pd.DataFrame(rows).sort_values("Score", ascending=False) if rows \
          else pd.DataFrame(columns=["Symbol"])

    # NEW: earnings blackout filter — only hits API for survivors
    if len(out):
        print(f"Checking earnings dates for {len(out)} survivors...")
        earn_cache = load_earnings_cache()
        today = dt.date.today()
        keep = []
        for _, r in out.iterrows():
            yf_sym = r["Symbol"] + ".NS"
            ed = next_earnings_date(yf_sym, earn_cache)
            if ed and 0 <= (ed - today).days <= P["earnings_blackout_days"]:
                print(f"  [skip] {r['Symbol']}: earnings on {ed}")
                continue
            keep.append(r)
        out = pd.DataFrame(keep) if keep else pd.DataFrame(columns=out.columns)
        if len(out):
            out = out.sort_values("Score", ascending=False).reset_index(drop=True)
        save_earnings_cache(earn_cache)

    # NEW: score delta vs yesterday (read BEFORE overwriting latest_scan.csv)
    latest_path = os.path.join(OUT_DIR, "latest_scan.csv")
    if os.path.exists(latest_path) and len(out):
        try:
            prev = pd.read_csv(latest_path)[["Symbol", "Score"]].rename(
                columns={"Score": "PrevScore"})
            out = out.merge(prev, on="Symbol", how="left")
            out["PrevScore"] = out["PrevScore"].fillna(out["Score"])
        except Exception:
            out["PrevScore"] = out["Score"]
    elif len(out):
        out["PrevScore"] = out["Score"]
    if len(out):
        out["ScoreDelta"] = (out["Score"] - out["PrevScore"]).round(0).astype(int)

    # NEW: position sizing columns
    if len(out):
        risk_per_share = out["Entry"] - out["Stop"]
        risk_amount    = P["capital"] * P["risk_per_trade_pct"] / 100
        out["Qty"]          = (risk_amount / risk_per_share).apply(
            lambda q: int(q) if q >= 1 else 0)
        out["CapitalUsed"]  = (out["Qty"] * out["Entry"]).round(0)

    today = dt.date.today().isoformat()
    out.to_csv(os.path.join(OUT_DIR, f"scan_{today}.csv"), index=False)
    out.to_csv(latest_path, index=False)
    ind_table.to_csv(os.path.join(OUT_DIR, "sector_strength.csv"), index=False)

    write_report(out, ind_table, n_last, mkt_note, mkt_green, breadth, today)
    write_markdown(out, ind_table, mkt_note, today, breadth, len(frames))

    # NEW: run log
    log_path = os.path.join(OUT_DIR, "run_log.csv")
    log_row = pd.DataFrame([{
        "date": today,
        "nifty_close": round(float(n_last["Close"]), 0),
        "breadth_pct": round(breadth, 0),
        "banner": "GREEN" if mkt_green else "RED",
        "n_setups": len(out),
        "top_score": float(out["Score"].max()) if len(out) else 0,
    }])
    log_row.to_csv(log_path, mode="a",
                   header=not os.path.exists(log_path), index=False)

    # NEW: positions check
    alerts = check_positions(frames)

    print(f"\n{mkt_note}\nBreadth: {breadth:.0f}% of Nifty 200 above 50-DMA")
    print(f"Qualified setups: {len(out)}  ->  output/report.html")
    if alerts:
        print("\nPosition alerts:")
        for a in alerts:
            print(f"  {a}")
    if len(out):
        cols = ["Symbol", "Entry", "Stop", "Target", "Risk_pct", "RR",
                "RS_3m", "Score", "ScoreDelta", "Qty"]
        print(out[cols].head(12).to_string(index=False))

def write_markdown(out, ind_table, mkt_note, today, breadth, n_scanned):
    lines = [f"# 📡 Swing Scan — {today}", "",
             f"**Market filter:** {mkt_note}  ",
             f"**Breadth:** {breadth:.0f}% of {n_scanned} stocks above 50-DMA",
             "", "## 🎯 Candidates (score-ranked)", ""]
    if len(out):
        lines += ["| Stock | Setup | Entry | Stop | Target | R:R | Risk% | RS 3m | ΔScore | Qty | Score |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in out.itertuples():
            delta = getattr(r, "ScoreDelta", 0)
            dstr  = f"+{delta}" if delta > 0 else str(delta)
            qty   = getattr(r, "Qty", 0)
            lines.append(
                f"| **{r.Symbol}** | {r.Setup} | {r.Entry:,.1f} | {r.Stop:,.1f} "
                f"| {r.Target:,.1f} | 1:{r.RR} | {r.Risk_pct}% | {r.RS_3m:+.1f}% "
                f"| {dstr} | {qty} | {r.Score:.0f} |")
    else:
        lines.append("_No qualifying setups today — cash is a position._")
    lines += ["", "## 🏭 Top industries (median 3-mo return)", ""]
    for x in ind_table.head(5).itertuples():
        lines.append(f"- {x.Industry}: **{x.MedianRet63:+.1f}%**")
    lines += ["", "---",
              f"_Position sizing assumes ₹{P['capital']:,} capital at "
              f"{P['risk_per_trade_pct']}% risk per trade. Set alerts at Entry; "
              "buy only on volume; verify results dates. Educational tool, not advice._"]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

def write_report(out, ind_table, n_last, mkt_note, mkt_green, breadth, today):
    badge = "g" if mkt_green else "r"

    rows_html = ""
    for _, r in out.iterrows():
        delta = int(r.get("ScoreDelta", 0))
        qty   = int(r.get("Qty", 0))
        cap   = float(r.get("CapitalUsed", 0))

        cls = "fresh" if str(r.Setup).endswith("breakout") else \
              ("trig" if "trigger" in r.Setup else "pb")
        if delta < 0:
            cls += " weak"

        dstr = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "0")

        rows_html += f"""<tr class="{cls}">
<td><b>{r.Symbol}</b><span class="co">{r.Company}</span></td>
<td>{r.Industry}</td><td><span class="tag {cls.split()[0]}">{r.Setup}</span></td>
<td>{r.Close:,.1f}</td><td class="e">{r.Entry:,.1f}</td><td class="s">{r.Stop:,.1f}</td>
<td class="t">{r.Target:,.1f}</td><td>{r.Risk_pct}%</td><td><b>1:{r.RR}</b></td>
<td>{'+' if r.RS_3m>0 else ''}{r.RS_3m}%</td><td>{r.ATR_pct}%</td><td>{r.Turnover_cr:,.0f}</td>
<td>{r.TopSector}</td><td>{dstr}</td><td>{qty}</td>
<td>{cap:,.0f}</td><td class="score">{r.Score:.0f}</td></tr>\n"""

    if not rows_html:
        rows_html = ('<tr><td colspan="17" class="none">'
                     'No qualifying setups today. Cash is a position.</td></tr>')

    sect = "".join(f"<tr><td>{x.Industry}</td><td>{x.MedianRet63:+.1f}%</td>"
                   f"<td>{x.N:.0f}</td></tr>"
                   for x in ind_table.head(8).itertuples())

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Swing Scan {today}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:24px;background:#0f1420;color:#e8ecf4}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#94a3b8;font-size:13px;margin-bottom:16px}}
.banner{{padding:12px 16px;border-radius:10px;font-weight:600;margin:12px 0}}
.banner.g{{background:#052e1b;border:1px solid #166534;color:#4ade80}}
.banner.r{{background:#2b0d0d;border:1px solid #991b1b;color:#f87171}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.card{{background:#1a2235;border:1px solid #2b3654;border-radius:10px;padding:12px 18px;min-width:130px}}
.card b{{display:block;font-size:20px}} .card span{{color:#94a3b8;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}}
th{{text-align:left;color:#94a3b8;font-size:10px;text-transform:uppercase;padding:6px;border-bottom:1px solid #2b3654}}
td{{padding:6px;border-bottom:1px solid #1e2942;vertical-align:top}}
tr.fresh td{{background:#10241c}} tr.trig td{{background:#161f33}} tr.pb td{{background:#231d10}}
tr.weak td{{background:#2b1a1a}}
.co{{display:block;color:#64748b;font-size:10px;font-weight:400}}
.e{{color:#60a5fa}} .s{{color:#f87171}} .t{{color:#4ade80}} .score{{font-weight:700;color:#facc15}}
.tag{{font-size:10px;padding:2px 6px;border-radius:6px;white-space:nowrap}}
.tag.fresh{{background:#166534;color:#fff}} .tag.trig{{background:#1d4ed8;color:#fff}} .tag.pb{{background:#a16207;color:#fff}}
.none{{text-align:center;padding:24px;color:#94a3b8}}
h2{{font-size:15px;margin-top:26px}} .foot{{margin-top:26px;color:#64748b;font-size:12px;line-height:1.6}}
.two{{display:flex;gap:24px;flex-wrap:wrap}} .box{{flex:1;min-width:280px}}
</style></head><body>
<h1>📡 Nifty 200 Swing Scanner</h1>
<div class="sub">Run date: {today} · Cash-market swing setups · R:R filter ≥ 1:{P['min_rr']:.0f}
 · Sizing: ₹{P['capital']:,} @ {P['risk_per_trade_pct']}% risk</div>
<div class="banner {badge}">Market filter: {mkt_note}</div>
<div class="cards">
<div class="card"><b>{n_last['Close']:,.0f}</b><span>Nifty close</span></div>
<div class="card"><b>{'Yes' if n_last['Close']>n_last['SMA50'] else 'No'}</b><span>Above 50-DMA</span></div>
<div class="card"><b>{breadth:.0f}%</b><span>of N200 above 50-DMA</span></div>
<div class="card"><b>{len(out)}</b><span>qualifying setups</span></div></div>
<div class="two"><div class="box">
<h2>🎯 Trade Candidates (ranked by score)</h2>
<table>
<tr><th>Stock</th><th>Industry</th><th>Setup</th><th>Close</th><th>Entry</th><th>Stop</th>
<th>Target</th><th>Risk%</th><th>R:R</th><th>RS 3m</th><th>ATR%</th><th>Turn ₹Cr</th>
<th>Top Sect</th><th>ΔScore</th><th>Qty</th><th>Capital</th><th>Score</th></tr>
{rows_html}</table></div>
<div class="box" style="max-width:300px"><h2>🏭 Strong industries (median 3-mo)</h2>
<table><tr><th>Industry</th><th>3-mo ret</th><th>#stk</th></tr>{sect}</table></div></div>
<div class="foot"><b>How to use:</b> Prioritise scores &gt; 60 in the top-3 industries.
For <b>Base near trigger</b>, set an alert at Entry — buy only on volume breakout.
For <b>Pullback</b>, Entry = above today's high (confirmation).
<b>ΔScore</b> compares vs yesterday; negative = weakening, be careful.
<b>Qty</b> is sized so a stop-out costs ~{P['risk_per_trade_pct']}% of ₹{P['capital']:,}.
Check upcoming results dates before entering. Never average down past the stop.<br><br>
<i>Educational tool. Not SEBI-registered investment advice. Verify prices before trading.</i></div>
</body></html>"""

    with open(os.path.join(OUT_DIR, "report.html"), "w") as f:
        f.write(html)

if __name__ == "__main__":
    main()