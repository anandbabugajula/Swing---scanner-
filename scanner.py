#!/usr/bin/env python3
"""
Nifty 200 Swing Trade Scanner (cash market, no F&O)
====================================================
Automates the 5-step funnel:
  1. Market filter      -> Nifty vs key moving averages + breadth
  2. Sector filter      -> strongest industries (median 3-mo return)
  3. Stock filters      -> trend, relative strength, ATR%, liquidity, near 52w high
  4. Setup detection    -> A) tight-base breakout   B) pullback to rising MA
  5. Trade plan         -> entry / stop / target / risk% / R:R + composite score

Outputs: output/scan_<date>.csv, output/latest_scan.csv, output/report.html

Run:  python3 scanner.py            (after market close, Mon-Fri)
Deps: pandas, numpy, yfinance, requests
"""

import os, sys, time, datetime as dt
import numpy as np
import pandas as pd
import requests

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "output")
CONS_CSV   = os.path.join(BASE_DIR, "nifty200_constituents.csv")
CONS_URL   = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"

# ----------------------------- tunable parameters -----------------------------
P = dict(
    history_days      = 400,     # lookback for indicators
    min_turnover_cr   = 30,      # 20d avg daily turnover floor (INR crore)
    atr_pct_min       = 2.0,     # volatility floor
    atr_pct_max       = 6.0,     # volatility ceiling
    max_dist_52w_high = 0.20,    # must be within 20% of 52-week high
    base_lookback     = 15,      # sessions defining a "base"
    base_tightness    = 0.12,    # base height <= 12% = tight
    trig_proximity    = 0.04,    # close within 4% of base high = near trigger
    breakout_vol_mult = 1.5,     # fresh breakout needs 1.5x volume
    pb_window         = 40,      # sessions in which the swing high must sit
    pb_depth_min      = 0.04,    # pullback depth 4%..
    pb_depth_max      = 0.12,    # ..to 12% off recent high
    pb_ma_tol         = 0.03,    # close within 3% of rising 20/50 DMA
    min_risk_pct      = 0.025,   # ignore trades with stop closer than 2.5%
    min_rr            = 2.0,     # only show trades with R:R >= 2
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
    df = pd.read_csv(CONS_CSV)
    df = df.rename(columns=str.strip)
    df["YF"] = df["Symbol"].str.strip() + ".NS"
    return df[["Company Name", "Industry", "Symbol", "YF"]]

def download(tickers, period="400d") -> dict:
    """Batch download with retry; returns {ticker: ohlcv_df}."""
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
        print(f"[info] skipped {len(failed)} tickers (no/insufficient data): "
              f"{', '.join(x.replace('.NS','') for x in failed[:8])}...")
    return out

# ------------------------------- indicators ----------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["ATR"]  = tr.ewm(alpha=1/14, adjust=False).mean()
    df["ATRp"] = df["ATR"] / c * 100
    df["V20"]  = v.rolling(20).mean()
    df["TURNcr"] = (c * v).rolling(20).mean() / 1e7      # avg daily turnover, INR cr
    df["RET63"] = c.pct_change(63) * 100                  # ~3-month momentum
    df["RET21"] = c.pct_change(21) * 100
    df["H52"]  = h.rolling(252, min_periods=200).max()
    return df

# ------------------------------- setups ---------------------------------------
def analyze(sym, name, ind, df, nifty_ret63) -> dict | None:
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
    if rs63 <= 0 or pd.isna(h52) or close < (1 - P["max_dist_52w_high"]) * h52:
        return None
    if not (P["atr_pct_min"] <= atrp <= P["atr_pct_max"]):
        return None
    if turn < P["min_turnover_cr"]:
        return None

    lb  = P["base_lookback"]
    win = df.iloc[-lb:]
    base_hi, base_lo = win["High"].max(), win["Low"].min()
    tight = (base_hi / base_lo - 1) <= P["base_tightness"]
    near_trig = close >= base_hi * (1 - P["trig_proximity"])
    v_ratio   = x["Volume"] / x["V20"] if x["V20"] else 0
    prior_hi  = win["High"].iloc[:-1].max()
    fresh_break = (close > prior_hi) and v_ratio >= P["breakout_vol_mult"]
    vol_dryup = df["Volume"].iloc[-5:].mean() < x["V20"]            # base contraction

    hi60      = df["High"].iloc[-P["pb_window"]:].max()
    hi60_pos  = df["High"].iloc[-P["pb_window"]:].idxmax()
    depth     = hi60 / close - 1
    sw_lo     = df["Low"].iloc[-10:].min()
    pb_ma_ok  = (abs(close/sma20 - 1) <= P["pb_ma_tol"] and sma20 > df["SMA20"].iloc[-6]) or \
                (abs(close/sma50 - 1) <= P["pb_ma_tol"] and sma50 > df["SMA50"].iloc[-6])
    bounce    = close > df["Close"].iloc[-2]                         # green day vs prior

    setup, entry, stop, target = None, None, None, None

    # ---- Setup A: breakout ----
    if fresh_break:
        setup = "A: Fresh breakout"
        entry = close
        stop  = max(base_lo, entry * 0.93)                          # base low or -7%
    elif tight and near_trig:
        setup = "A: Base near trigger"
        entry = round(base_hi, 2)
        stop  = max(base_lo, entry * 0.93)
    # ---- Setup B: pullback ----
    if setup is None or setup.startswith("A"):
        if (P["pb_depth_min"] <= depth <= P["pb_depth_max"] and pm_ok(pb_ma_ok)
                and bounce and (df.index[-1] > hi60_pos)):
            e = round(df["High"].iloc[-1] * 1.002, 2)               # above today's high
            s = round(sw_lo * 0.995, 2)                             # swing low buffer
            t = hi60
            if s < e and (e - s) / e >= P["min_risk_pct"]:
                rr = (t - e) / (e - s)
                if rr >= P["min_rr"] and (setup is None or s > stop):
                    if setup is None:
                        setup, entry, stop = "B: Pullback to MA", e, s
                        target = t

    if setup is None or entry is None:
        return None
    risk_pct = (entry - stop) / entry * 100
    if risk_pct < P["min_risk_pct"] * 100:
        return None
    if target is None:                                              # breakout: measured move
        target = round(entry * (1 + 2.5 * risk_pct / 100), 2)
    rr = (target - entry) / (entry - stop)

    # ---- composite score (0-100) ----
    score = 15 + min(rs63, 30) / 30 * 30                            # relative strength
    if setup.startswith("A"):
        score += (1 - (base_hi/base_lo - 1)) * 25                   # base tightness
        score += 10 if vol_dryup else 0
        score += 15 if fresh_break else max(0, 15 * (1 - (base_hi/close - 1)/P["trig_proximity"]))
    else:
        score += 15 * (1 - abs(depth - 0.06) / 0.06)                # ideal 6% dip
        score += 10 if pm_ok(pb_ma_ok) else 0
    score += min(max((rr - 2), 0), 1) * 10                          # rr quality
    in_top_sector = ind in TOP_SECTORS
    score += 10 if in_top_sector else 0

    return dict(Symbol=sym, Company=name, Industry=ind, Setup=setup,
                Close=round(close, 2), Entry=entry, Stop=round(stop, 2),
                Target=round(target, 2), Risk_pct=round(risk_pct, 1),
                RR=round(rr, 2), RS_3m=round(rs63, 1), ATR_pct=round(atrp, 1),
                Vol_x=round(v_ratio, 1), Turnover_cr=round(turn, 0),
                TopSector="Yes" if in_top_sector else "-",
                Score=round(min(score, 100), 0))

def pm_ok(v):  # tiny helper so setup blocks stay readable
    return bool(v)

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

    # sector strength = median 3-mo return by industry
    ind_ret = {}
    for t, d in frames.items():
        r = d.iloc[-1]["RET63"]
        if not pd.isna(r):
            ind_ret.setdefault(cons.loc[cons["YF"] == t, "Industry"].iloc[0], []).append(r)
    ind_table = (pd.DataFrame([(k, np.median(v), len(v)) for k, v in ind_ret.items()],
                              columns=["Industry", "MedianRet63", "N"])
                   .sort_values("MedianRet63", ascending=False))
    global TOP_SECTORS
    TOP_SECTORS = set(ind_table.head(5)["Industry"])

    nifty_ret63 = n_last["RET63"]
    rows = []
    for _, r in cons.iterrows():
        if r["YF"] not in frames:
            continue
        try:
            res = analyze(r["Symbol"], r["Company Name"], r["Industry"],
                          frames[r["YF"]], nifty_ret63)
            if res:
                rows.append(res)
        except Exception as e:
            print(f"[warn] {r['Symbol']}: {e}")

    out = pd.DataFrame(rows).sort_values("Score", ascending=False) if rows \
          else pd.DataFrame(columns=["Symbol"])
    today = dt.date.today().isoformat()
    out.to_csv(os.path.join(OUT_DIR, f"scan_{today}.csv"), index=False)
    out.to_csv(os.path.join(OUT_DIR, "latest_scan.csv"), index=False)
    ind_table.to_csv(os.path.join(OUT_DIR, "sector_strength.csv"), index=False)

    write_report(out, ind_table, n_last, mkt_note, mkt_green, breadth, today)
    write_markdown(out, ind_table, mkt_note, today, breadth, len(frames))
    print(f"\n{mkt_note}\nBreadth: {breadth:.0f}% of Nifty 200 above 50-DMA")
    print(f"Qualified setups: {len(out)}  ->  output/report.html")
    if len(out):
        print(out.head(12).to_string(index=False))

def write_markdown(out, ind_table, mkt_note, today, breadth, n_scanned):
    """Phone-friendly report: GitHub renders .md natively in the repo view."""
    lines = [f"# 📡 Swing Scan — {today}", "",
             f"**Market filter:** {mkt_note}  ",
             f"**Breadth:** {breadth:.0f}% of {n_scanned} stocks above 50-DMA",
             "", "## 🎯 Candidates (score-ranked)", ""]
    if len(out):
        lines += ["| Stock | Setup | Entry | Stop | Target | R:R | Risk% | RS 3m | Score |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in out.itertuples():
            lines.append(f"| **{r.Symbol}** | {r.Setup} | {r.Entry:,.1f} | {r.Stop:,.1f} "
                         f"| {r.Target:,.1f} | 1:{r.RR} | {r.Risk_pct}% | {r.RS_3m:+.1f}% | {r.Score:.0f} |")
    else:
        lines.append("_No qualifying setups today — cash is a position._")
    lines += ["", "## 🏭 Top industries (median 3-mo return)", ""]
    for x in ind_table.head(5).itertuples():
        lines.append(f"- {x.Industry}: **{x.MedianRet63:+.1f}%**")
    lines += ["", "---", "_Set alerts at Entry for 'Base near trigger' rows; verify results dates "
              "before entering; risk max 1–2% of capital per trade. Educational tool, not advice._"]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))


def write_report(out, ind_table, n_last, mkt_note, mkt_green, breadth, today):
    def badge(s): return "g" if s else "r"
    rows_html = ""
    for _, r in out.iterrows():
        cls = "fresh" if str(r.Setup).endswith("breakout") else \
              ("trig" if "trigger" in r.Setup else "pb")
        rows_html += f"""<tr class="{cls}">
<td><b>{r.Symbol}</b><span class="co">{r.Company}</span></td>
<td>{r.Industry}</td><td><span class="tag {cls}">{r.Setup}</span></td>
<td>{r.Close:,.1f}</td><td class="e">{r.Entry:,.1f}</td><td class="s">{r.Stop:,.1f}</td>
<td class="t">{r.Target:,.1f}</td><td>{r.Risk_pct}%</td><td><b>1:{r.RR}</b></td>
<td>{'+' if r.RS_3m>0 else ''}{r.RS_3m}%</td><td>{r.ATR_pct}%</td><td>{r.Turnover_cr:,.0f}</td>
<td>{r.TopSector}</td><td class="score">{r.Score:.0f}</td></tr>\n"""
    if not rows_html:
        rows_html = '<tr><td colspan="14" class="none">No qualifying setups today. Cash is a position.</td></tr>'
    sect = "".join(f"<tr><td>{x.Industry}</td><td>{x.MedianRet63:+.1f}%</td><td>{x.N:.0f}</td></tr>"
                   for x in ind_table.head(8).itertuples())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Swing Scan {today}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:24px;background:#0f1420;color:#e8ecf4}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#94a3b8;font-size:13px;margin-bottom:16px}}
.banner{{padding:12px 16px;border-radius:10px;font-weight:600;margin:12px 0}}
.banner.g{{background:#052e1b;border:1px solid #166534;color:#4ade80}}
.banner.r{{background:#2b0d0d;border:1px solid #991b1b;color:#f87171}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.card{{background:#1a2235;border:1px solid #2b3654;border-radius:10px;padding:12px 18px;min-width:130px}}
.card b{{display:block;font-size:20px}} .card span{{color:#94a3b8;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}}
th{{text-align:left;color:#94a3b8;font-size:11px;text-transform:uppercase;padding:8px;border-bottom:1px solid #2b3654}}
td{{padding:8px;border-bottom:1px solid #1e2942;vertical-align:top}}
tr.fresh td{{background:#10241c}} tr.trig td{{background:#161f33}} tr.pb td{{background:#231d10}}
.co{{display:block;color:#64748b;font-size:11px;font-weight:400}}
.e{{color:#60a5fa}} .s{{color:#f87171}} .t{{color:#4ade80}} .score{{font-weight:700;color:#facc15}}
.tag{{font-size:11px;padding:2px 8px;border-radius:8px;white-space:nowrap}}
.tag.fresh{{background:#166534;color:#fff}} .tag.trig{{background:#1d4ed8;color:#fff}} .tag.pb{{background:#a16207;color:#fff}}
.none{{text-align:center;padding:24px;color:#94a3b8}}
h2{{font-size:15px;margin-top:26px}} .foot{{margin-top:26px;color:#64748b;font-size:12px;line-height:1.6}}
.two{{display:flex;gap:24px;flex-wrap:wrap}} .box{{flex:1;min-width:280px}}
</style></head><body>
<h1>📡 Nifty 200 Swing Scanner</h1><div class="sub">Run date: {today} · Cash-market swing setups only · R:R filter ≥ 1:{P['min_rr']:.0f}</div>
<div class="banner {badge(mkt_green)}">Market filter: {mkt_note}</div>
<div class="cards">
<div class="card"><b>{n_last['Close']:,.0f}</b><span>Nifty close</span></div>
<div class="card"><b>{'Yes' if n_last['Close']>n_last['SMA50'] else 'No'}</b><span>Above 50-DMA</span></div>
<div class="card"><b>{breadth:.0f}%</b><span>of N200 above 50-DMA</span></div>
<div class="card"><b>{len(out)}</b><span>qualifying setups</span></div></div>
<div class="two"><div class="box">
<h2>🎯 Trade Candidates (ranked by score)</h2>
<table><tr><th>Stock</th><th>Industry</th><th>Setup</th><th>Close</th><th>Entry</th><th>Stop</th>
<th>Target</th><th>Risk%</th><th>R:R</th><th>RS 3m</th><th>ATR%</th><th>Turn ₹Cr</th><th>Top Sect</th><th>Score</th></tr>
{rows_html}</table></div>
<div class="box" style="max-width:300px"><h2>🏭 Strong industries (median 3-mo)</h2>
<table><tr><th>Industry</th><th>3-mo ret</th><th>#stk</th></tr>{sect}</table></div></div>
<div class="foot"><b>How to use:</b> Prioritise scores &gt; 60 in the top-3 industries. For
<b>Base near trigger</b> rows, set an alert at the Entry price — buy only on breakout with volume.
For <b>Pullback</b> rows, Entry = above today's high (confirmation). Risk 1–2% of capital per trade,
check upcoming results dates before entering, and never average down past the stop.<br><br>
<i>Educational tool. Not SEBI-registered investment advice. Verify prices before trading.</i></div>
</body></html>"""
    with open(os.path.join(OUT_DIR, "report.html"), "w") as f:
        f.write(html)

if __name__ == "__main__":
    main()
