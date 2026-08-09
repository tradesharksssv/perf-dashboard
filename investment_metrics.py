"""
investment_metrics.py
---------------------
Calculate portfolio evaluation metrics and generate an HTML dashboard.
Usage:
    python investment_metrics.py portfolio_data.csv
    python investment_metrics.py portfolio_data.csv my_dashboard.html
Expected columns (tab- or comma-separated):
    date, overallPNL, PNLonFundsChurned, totalAccountValue,
    totalIdleFund, percIdleFund, percPNL_onHoldingFund,
    niftyReturns, sensexReturns, niftyBankReturns,
    niftyMidCapReturns, niftySmallCapReturns
Requires: pandas, numpy  (pip install pandas numpy)
"""
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)
    df = _drop_index_column(df)
    df = _parse_date_column(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a leading integer-index column that pandas or Excel sometimes writes."""
    first = str(df.columns[0]).strip()
    if first in ("", "Unnamed: 0") or first.isdigit():
        df = df.iloc[:, 1:]
    elif pd.api.types.is_integer_dtype(df.iloc[:, 0]):
        expected = pd.Series(range(len(df)))
        if (df.iloc[:, 0].reset_index(drop=True) == expected).all():
            df = df.iloc[:, 1:]
    return df.reset_index(drop=True)


def _parse_date_column(df):
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        print(f"Warning: {n_bad} date value(s) could not be parsed.")
    df = df.dropna(subset=["date"])
    return df


# ---------------------------------------------------------------------------
# 2. Metric calculations
# ---------------------------------------------------------------------------

def _safe_mean(series: pd.Series):
    s = series.dropna()
    return float(s.mean()) if len(s) else None


def _safe_last(series: pd.Series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def calculate_metrics(df: pd.DataFrame) -> dict:
    m = {}
    pnl = df["overallPNL"].dropna()

    # --- Absolute return --------------------------------------------------
    m["total_pnl"]     = ((1 + pnl / 100).prod() - 1) * 100
    m["avg_daily_pnl"] = float(pnl.mean()) if len(pnl) else None
    m["latest_pnl"]    = float(pnl.iloc[-1]) if len(pnl) else None

    hold = df.get("percPNL_onHoldingFund", pd.Series(dtype=float))
    m["avg_return_on_holding"]    = _safe_mean(hold)
    m["latest_return_on_holding"] = _safe_last(hold)

    churned = df.get("PNLonFundsChurned", pd.Series(dtype=float))
    mask = churned.notna() & pnl.notna() & (pnl != 0)
    m["avg_churn_efficiency"] = float((churned[mask] / pnl[mask]).mean()) if mask.sum() else None

    # --- Alpha vs benchmarks ----------------------------------------------
    benchmarks = {
        "nifty":         "niftyReturns",
        "sensex":        "sensexReturns",
        "nifty_bank":    "niftyBankReturns",
        "nifty_midcap":  "niftyMidCapReturns",
        "nifty_smallcap":"niftySmallCapReturns",
    }
    alpha = {}
    strategy_return = ((1 + pnl / 100).prod() - 1) * 100
    for label, col in benchmarks.items():
        if col not in df.columns:
            continue
        bm = df[col].dropna()
        if len(bm) == 0:
            continue
        bm_return = ((1 + bm / 100).prod() - 1) * 100
        alpha[label] = round(strategy_return - bm_return, 2)

    m["strategy_return"] = round(strategy_return, 2)
    m["alpha"] = alpha

    # --- Mutual fund alpha ------------------------------------------------
    mutual_funds = {
        "Parag Parikh Flexi": "Parag_Parikh_Flexi_Cap_Fund",
        "Quant MF":           "QUANTMUTUALFUND_MF",
        "SBI MF":             "SBIMutualFund_MF",
        "Motilal Oswal MF":   "MOTILALOSWAL_MF",
        "Bandhan MF":         "BANDHANMUTUALFUND_MF",
        "Edelweiss MF":       "EDELWEISSMUTUALFUND_MF",
        "HDFC MF":            "HDFCMutualFund_MF",
        "Invesco MidCap MF":  "INVESCOMUTUALFUND_MidCap_MF",
    }
    mf_alpha = {}
    for label, col in mutual_funds.items():
        if col not in df.columns:
            continue
        bm = df[col].dropna()
        if len(bm) == 0:
            continue
        bm_return = ((1 + bm / 100).prod() - 1) * 100
        mf_alpha[label] = round(strategy_return - bm_return, 2)
    m["mf_alpha"] = mf_alpha

    # --- Capital efficiency -----------------------------------------------
    idle_pct = df.get("percIdleFund", pd.Series(dtype=float))
    m["avg_idle_pct"]    = _safe_mean(idle_pct)
    m["latest_idle_pct"] = _safe_last(idle_pct)
    m["avg_utilization"] = (100 - _safe_mean(idle_pct)) if _safe_mean(idle_pct) is not None else None

    idle_fund = df.get("totalIdleFund", pd.Series(dtype=float))
    m["latest_idle_fund"] = _safe_last(idle_fund)

    if {"totalAccountValue", "totalIdleFund", "overallPNL"}.issubset(df.columns):
        vm = df["totalAccountValue"].notna() & df["totalIdleFund"].notna() & pnl.notna()
        if vm.sum():
            deployed = (df.loc[vm, "totalAccountValue"] - df.loc[vm, "totalIdleFund"]).replace(0, np.nan)
            m["return_per_deployed"] = float((pnl[vm] / deployed).mean())
    else:
        m["return_per_deployed"] = None

    # --- Risk metrics -----------------------------------------------------
    if len(pnl) > 1:
        m["daily_pnl_volatility"] = float(pnl.std())

        equity      = 100 * (1 + pnl / 100).cumprod()
        rolling_max = equity.cummax()
        drawdown    = (equity / rolling_max - 1) * 100

        m["max_drawdown"] = float(drawdown.min())
        m["win_rate"]     = float((pnl > 0).mean())
        m["loss_rate"]    = float((pnl < 0).mean())
        m["flat_rate"]    = float((pnl == 0).mean())

        if m["daily_pnl_volatility"] and m["daily_pnl_volatility"] != 0:
            m["sharpe_like"] = float(m["avg_daily_pnl"] / m["daily_pnl_volatility"])
        else:
            m["sharpe_like"] = None
    else:
        m["daily_pnl_volatility"] = m["max_drawdown"] = None
        m["win_rate"] = m["loss_rate"] = m["flat_rate"] = m["sharpe_like"] = None

    # --- Trend ------------------------------------------------------------
    m["rolling_7d_pnl"] = float(pnl.rolling(7).mean().iloc[-1]) if len(pnl) >= 7 else None

    df2 = df.copy()
    df2["month"] = df2["date"].dt.to_period("M")
    monthly = (
        df2.groupby("month")["overallPNL"]
           .apply(lambda x: ((1 + x / 100).prod() - 1) * 100)
    )
    m["monthly_pnl"] = {str(k): round(v, 6) for k, v in monthly.items()}
    m["mom_change"]  = float(monthly.iloc[-1] - monthly.iloc[-2]) if len(monthly) >= 2 else None

    return m


# ---------------------------------------------------------------------------
# 3. Terminal summary
# ---------------------------------------------------------------------------

def print_summary(m: dict):
    def row(label, val, pct=False, already_pct=False, higher_good=True):
        if val is None:
            print(f"  {label:<40}  {'N/A':>12}")
            return
        if already_pct:
            fval = f"{val:.2f}%"
        elif pct:
            fval = f"{val * 100:.2f}%"
        else:
            fval = f"{val:+.4f}"
        mark = "V" if (val > 0) == higher_good else "X"
        print(f"  {label:<40}  {fval:>12}  {mark}")

    print("\n" + "=" * 60)
    print("  PORTFOLIO METRICS SUMMARY")
    print("=" * 60)

    print("\n  RETURN PERFORMANCE")
    row("Compounded total return (%)", m.get("total_pnl"))
    row("Avg daily P&L",               m.get("avg_daily_pnl"))
    row("Return on holding fund (avg)", m.get("avg_return_on_holding"))
    row("Churn efficiency ratio",       m.get("avg_churn_efficiency"))

    print("\n  BENCHMARK ALPHA")
    for k, v in (m.get("alpha") or {}).items():
        row(f"Alpha vs {k.replace('_', ' ').title()}", v)

    print("\n  CAPITAL EFFICIENCY")
    row("Avg capital utilization", m.get("avg_utilization"), already_pct=True)
    row("Avg idle fund %",         m.get("avg_idle_pct"),    already_pct=True, higher_good=False)
    row("Return per Rupee deployed", m.get("return_per_deployed"))

    print("\n  RISK METRICS")
    row("Win rate",             m.get("win_rate"),              pct=True)
    row("Loss rate",            m.get("loss_rate"),             pct=True, higher_good=False)
    row("Flat rate",            m.get("flat_rate"),             pct=True, higher_good=False)
    row("Max drawdown",         m.get("max_drawdown"),          higher_good=False)
    row("Daily P&L volatility", m.get("daily_pnl_volatility"), higher_good=False)
    row("Sharpe-like ratio",    m.get("sharpe_like"))

    print("\n  TREND")
    row("7-day rolling avg P&L",   m.get("rolling_7d_pnl"))
    row("Month-over-month change", m.get("mom_change"))
    print("=" * 60 + "\n")


def calc_drawdown(series):
    peak = series.cummax()
    return (series / peak - 1) * 100


# ---------------------------------------------------------------------------
# 4. HTML dashboard generation
# ---------------------------------------------------------------------------

def generate_dashboard(df: pd.DataFrame, m: dict, output_path: str = "investment_dashboard.html"):
    df_c = df.dropna(subset=["overallPNL"]).copy()

    # Equity curves (Growth of Rs.100)
    df_c["strategy_equity"] = 100 * (1 + df_c["overallPNL"] / 100).cumprod()

    benchmarks = {
        "Nifty":      "niftyReturns",
        "Sensex":     "sensexReturns",
        "Bank Nifty": "niftyBankReturns",
        "Midcap":     "niftyMidCapReturns",
        "Smallcap":   "niftySmallCapReturns",
    }
    for name, col in benchmarks.items():
        if col in df_c.columns:
            df_c[f"{name}_equity"] = 100 * (1 + df_c[col].fillna(0) / 100).cumprod()

    # Mutual fund equity curves
    mf_cols = {
        "Parag Parikh Flexi": "Parag_Parikh_Flexi_Cap_Fund",
        "Quant MF":           "QUANTMUTUALFUND_MF",
        "SBI MF":             "SBIMutualFund_MF",
        "Motilal Oswal MF":   "MOTILALOSWAL_MF",
        "Bandhan MF":         "BANDHANMUTUALFUND_MF",
        "Edelweiss MF":       "EDELWEISSMUTUALFUND_MF",
        "HDFC MF":            "HDFCMutualFund_MF",
        "Invesco MidCap MF":  "INVESCOMUTUALFUND_MidCap_MF",
        "ICICI MF" : "ICICIPrudentialMutualFund_MF",
        "Birla SunLife MF" : "BirlaSunLifeMutualFund_MF",
        "Invesco MF": "INVESCOMUTUALFUND_MF"
    }
    active_mfs = {name: col for name, col in mf_cols.items() if col in df_c.columns}
    for name, col in active_mfs.items():
        df_c[f"mf_{name}_equity"] = 100 * (1 + df_c[col].fillna(0) / 100).cumprod()

    # Benchmark scorecard
    strategy_final = df_c["strategy_equity"].iloc[-1]
    benchmark_summary = []
    for name in benchmarks:
        bm_final = df_c[f"{name}_equity"].iloc[-1]
        benchmark_summary.append({
            "Benchmark": name,
            "Return %":  round((bm_final / 100 - 1) * 100, 2),
            "Alpha %":   round((strategy_final / bm_final - 1) * 100, 2),
            "Max DD %":  round(calc_drawdown(df_c[f"{name}_equity"]).min(), 2),
        })

    strategy_return   = round((strategy_final / 100 - 1) * 100, 2)
    strategy_dd       = round(calc_drawdown(df_c["strategy_equity"]).min(), 2)

    strategy_equity   = df_c["strategy_equity"].round(2).tolist()
    nifty_equity      = df_c["Nifty_equity"].round(2).tolist()
    sensex_equity     = df_c["Sensex_equity"].round(2).tolist()
    bank_equity       = df_c["Bank Nifty_equity"].round(2).tolist()
    midcap_equity     = df_c["Midcap_equity"].round(2).tolist()
    smallcap_equity   = df_c["Smallcap_equity"].round(2).tolist()

    strategy_dd_curve = calc_drawdown(df_c["strategy_equity"]).round(2).tolist()
    nifty_dd_curve    = calc_drawdown(df_c["Nifty_equity"]).round(2).tolist()
    midcap_dd_curve   = calc_drawdown(df_c["Midcap_equity"]).round(2).tolist()
    smallcap_dd_curve = calc_drawdown(df_c["Smallcap_equity"]).round(2).tolist()

    benchmark_rows = ""
    for row in benchmark_summary:
        alpha_cls = "good" if row["Alpha %"] > 0 else "bad"
        benchmark_rows += (
            f"<tr><td>{row['Benchmark']}</td>"
            f"<td>{row['Return %']}%</td>"
            f"<td class='{alpha_cls}'>{row['Alpha %']}%</td>"
            f"<td>{row['Max DD %']}%</td></tr>\n"
        )

    df_c["cumulative_pnl"] = 100 * (1 + df_c["overallPNL"] / 100).cumprod()
    dates        = [d.strftime("%Y-%m-%d") for d in df_c["date"]]
    cum_pnl      = df_c["cumulative_pnl"].round(2).tolist()
    daily_pnl    = df_c["overallPNL"].round(6).tolist()

    monthly_labels = list(m["monthly_pnl"].keys())
    monthly_vals   = list(m["monthly_pnl"].values())

    idle_fund_vals = (df_c["totalIdleFund"].fillna(0).round(2).tolist()
                      if "totalIdleFund" in df_c.columns else [0] * len(df_c))
    idle_pct_vals  = (df_c["percIdleFund"].fillna(0).round(2).tolist()
                      if "percIdleFund"  in df_c.columns else [0] * len(df_c))

    # Mutual fund equity arrays + scorecard
    mf_names_list  = list(active_mfs.keys())
    mf_equities    = [df_c[f"mf_{n}_equity"].round(2).tolist() for n in mf_names_list]
    mf_dd_curves   = [calc_drawdown(df_c[f"mf_{n}_equity"]).round(2).tolist() for n in mf_names_list]

    mf_scorecard_rows = ""
    for name in mf_names_list:
        mf_final  = df_c[f"mf_{name}_equity"].iloc[-1]
        mf_ret    = round((mf_final / 100 - 1) * 100, 2)
        mf_alpha_v = round((strategy_final / mf_final - 1) * 100, 2)
        mf_dd_v   = round(calc_drawdown(df_c[f"mf_{name}_equity"]).min(), 2)
        alpha_cls = "good" if mf_alpha_v > 0 else "bad"
        mf_scorecard_rows += (
            f"<tr><td>{name}</td>"
            f"<td>{mf_ret}%</td>"
            f"<td class='{alpha_cls}'>{mf_alpha_v}%</td>"
            f"<td>{mf_dd_v}%</td></tr>\n"
        )

    bm_names = ["Nifty", "Sensex", "Nifty Bank", "Nifty Midcap", "Nifty Smallcap"]
    bm_keys  = ["nifty", "sensex", "nifty_bank", "nifty_midcap", "nifty_smallcap"]
    alpha_vals = [round(m["alpha"].get(k, 0), 6) for k in bm_keys]

    def fmt(val, pct=False, decimals=4):
        if val is None:
            return "N/A"
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:+.{decimals}f}" if val != 0 else "0.0000"

    def color(val, higher_good=True):
        if val is None:
            return "neutral"
        return "good" if (val > 0) == higher_good else "bad"

    alpha_rows = ""
    for name, val in zip(bm_names, alpha_vals):
        if not m["alpha"]:
            break
        badge_cls = "badge-good" if val >= 0 else "badge-bad"
        status    = "Outperforming" if val >= 0 else "Underperforming"
        val_cls   = "good" if val >= 0 else "bad"
        alpha_rows += (
            f"<tr><td>{name}</td>"
            f"<td class='{val_cls}'>{val:+.4f}</td>"
            f"<td><span class='badge {badge_cls}'>{status}</span></td></tr>\n"
        )
    if not alpha_rows:
        alpha_rows = "<tr><td colspan='3' style='color:#888780;font-size:13px'>Benchmark data not yet in dataset.</td></tr>"

    generated = datetime.now().strftime("%B %d, %Y at %H:%M")
    n_days    = len(df_c)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Investment dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f3;color:#1a1a18;padding:2rem;}}
h1{{font-size:22px;font-weight:500;margin-bottom:.25rem}}
.filter-bar{{display:flex;gap:8px;margin-bottom:1rem;flex-wrap:wrap}}
.filter-btn{{font-size:12px;padding:5px 14px;border:.5px solid #e0ded9;border-radius:20px;background:#fff;cursor:pointer;color:#636360}}
.filter-btn:hover{{background:#f5f5f3}}
.filter-btn.active{{background:#185FA5;color:#fff;border-color:#185FA5}}
.sub{{font-size:14px;color:#6b6b66;margin-bottom:2rem}}
.sec{{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;color:#888780;margin:1.75rem 0 .75rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.kpi{{background:#fff;border:.5px solid #e0ded9;border-radius:10px;padding:1rem}}
.kpi-label{{font-size:12px;color:#888780;margin-bottom:4px}}
.kpi-value{{font-size:22px;font-weight:500}}
.kpi-sub{{font-size:12px;color:#888780;margin-top:2px}}
.good{{color:#3B6D11}}.bad{{color:#A32D2D}}.neutral{{color:#636360}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin-top:.75rem}}
.chart-card{{background:#fff;border:.5px solid #e0ded9;border-radius:10px;padding:1.25rem}}
.chart-title{{font-size:14px;font-weight:500;margin-bottom:1rem}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-weight:500;font-size:12px;color:#888780;padding:6px 0;border-bottom:.5px solid #e0ded9}}
td{{padding:8px 0;border-bottom:.5px solid #f0ede8}}
.badge{{font-size:11px;padding:2px 8px;border-radius:4px;display:inline-block}}
.badge-good{{background:#EAF3DE;color:#3B6D11}}
.badge-bad{{background:#FCEBEB;color:#A32D2D}}
footer{{font-size:12px;color:#888780;margin-top:2rem;padding-top:1rem;border-top:.5px solid #e0ded9}}
</style>
</head>
<body>
<h1>Portfolio performance dashboard</h1>
<p class="sub">Generated {generated} &nbsp;&middot;&nbsp; {n_days} trading sessions</p>

<p class="sec">Return performance</p>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Total returns</div>
    <div class="kpi-value {color(m.get('total_pnl'))}">{fmt(m.get('total_pnl'))}</div>
    <div class="kpi-sub">Compounded across all days</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg daily P&amp;L</div>
    <div class="kpi-value {color(m.get('avg_daily_pnl'))}">{fmt(m.get('avg_daily_pnl'))}</div>
    <div class="kpi-sub">Mean per session</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win rate</div>
    <div class="kpi-value {color(m.get('win_rate'), higher_good=True)}">{fmt(m.get('win_rate'), pct=True)}</div>
    <div class="kpi-sub">% days with positive P&amp;L</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Return on holding fund</div>
    <div class="kpi-value {color(m.get('avg_return_on_holding'))}">{fmt(m.get('avg_return_on_holding'))}</div>
    <div class="kpi-sub">Avg % on deployed capital</div>
  </div>
</div>

<p class="sec">Risk metrics</p>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Max drawdown</div>
    <div class="kpi-value bad">{fmt(m.get('max_drawdown'))}</div>
    <div class="kpi-sub">Worst cumulative drop from peak</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Daily P&amp;L volatility</div>
    <div class="kpi-value neutral">{fmt(m.get('daily_pnl_volatility'))}</div>
    <div class="kpi-sub">Std dev of daily P&amp;L</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe-like ratio</div>
    <div class="kpi-value {color(m.get('sharpe_like'))}">{fmt(m.get('sharpe_like'))}</div>
    <div class="kpi-sub">Avg return / volatility</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Capital utilization</div>
    <div class="kpi-value neutral">{fmt(m.get('avg_utilization'))}</div>
    <div class="kpi-sub">Avg non-idle capital</div>
  </div>
</div>

<div class="filter-bar" id="filterBar">
  <button class="filter-btn" data-months="1">1M</button>
  <button class="filter-btn" data-months="3">3M</button>
  <button class="filter-btn" data-months="6">6M</button>
  <button class="filter-btn" data-months="ytd">YTD</button>
  <button class="filter-btn" data-months="12">1Y</button>
  <button class="filter-btn active" data-months="all">All</button>
</div>

<p class="sec">Charts</p>
<div class="chart-grid">
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Total returns over time</div>
    <div style="position:relative;height:220px">
      <canvas id="cChart" role="img" aria-label="Line chart of total returns over time">Total portfolio return trend.</canvas>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Daily P&amp;L (green = gain, red = loss)</div>
    <div style="position:relative;height:200px">
      <canvas id="dChart" role="img" aria-label="Bar chart of daily P&L">Daily P&L bar chart.</canvas>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Monthly P&amp;L</div>
    <div style="position:relative;height:200px">
      <canvas id="mChart" role="img" aria-label="Bar chart of monthly P&L">Monthly P&L aggregated by calendar month.</canvas>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Win / loss / flat days</div>
    <div style="position:relative;height:180px">
      <canvas id="winPieChart" role="img" aria-label="Doughnut chart showing win loss and flat day distribution">Trading day outcome distribution.</canvas>
    </div>
    <div id="pieLegend" style="display:flex;justify-content:center;gap:16px;margin-top:14px;font-size:12px;color:#636360;flex-wrap:wrap"></div>
  </div>

  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Idle fund &mdash; daily (Rs. amount &amp; % of portfolio)</div>
    <div style="position:relative;height:220px">
      <canvas id="idleChart" role="img" aria-label="Bar chart of daily idle fund amount with percentage overlay">Daily idle fund levels.</canvas>
    </div>
  </div>

  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Alpha vs benchmark indices</div>
    <table>
      <thead><tr><th>Index</th><th>Your alpha</th><th>Status</th></tr></thead>
      <tbody>{alpha_rows}</tbody>
    </table>
    <p style="font-size:12px;color:#888780;margin-top:.75rem">Alpha = compounded overallPNL return minus compounded benchmark return. Positive means the strategy outperformed the benchmark.</p>
  </div>
</div>

<p class="sec">Benchmark comparison</p>
<div class="chart-grid">
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Growth of Rs.100</div>
    <div style="position:relative;height:350px">
      <canvas id="benchmarkChart" role="img" aria-label="Line chart showing growth of 100 rupees for strategy vs benchmarks">Equity curve comparison.</canvas>
    </div>
  </div>
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Drawdown comparison</div>
    <div style="position:relative;height:350px">
      <canvas id="drawdownChart" role="img" aria-label="Line chart showing drawdown comparison across strategy and benchmarks">Drawdown curves.</canvas>
    </div>
  </div>
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Benchmark scorecard</div>
    <table>
      <thead>
        <tr><th>Benchmark</th><th>Total return</th><th>Alpha vs strategy</th><th>Max drawdown</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Strategy</b></td>
          <td><b>{strategy_return}%</b></td>
          <td>&mdash;</td>
          <td><b>{strategy_dd}%</b></td>
        </tr>
        {benchmark_rows}
      </tbody>
    </table>
  </div>
</div>

<p class="sec">Mutual fund comparison</p>
<div class="chart-grid">
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Growth of Rs.100 &mdash; mutual funds vs strategy</div>
    <div style="position:relative;height:350px">
      <canvas id="mfChart" role="img" aria-label="Line chart comparing mutual fund equity curves vs strategy">MF equity curve comparison.</canvas>
    </div>
  </div>
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Drawdown &mdash; mutual funds vs strategy</div>
    <div style="position:relative;height:280px">
      <canvas id="mfDDChart" role="img" aria-label="Line chart showing drawdown for mutual funds vs strategy">MF drawdown curves.</canvas>
    </div>
  </div>
  <div class="chart-card" style="grid-column:1/-1">
    <div class="chart-title">Mutual fund scorecard</div>
    <table>
      <thead>
        <tr><th>Fund</th><th>Total return</th><th>Alpha vs strategy</th><th>Max drawdown</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Strategy</b></td>
          <td><b>{strategy_return}%</b></td>
          <td>&mdash;</td>
          <td><b>{strategy_dd}%</b></td>
        </tr>
        {mf_scorecard_rows}
      </tbody>
    </table>
  </div>
</div>

<footer>
  All return and benchmark metrics are calculated using compounded returns from overallPNL and index return columns.
  Capital utilization uses fund deployment fields. N/A = insufficient data. &nbsp;&middot;&nbsp; Script: investment_metrics.py
</footer>

<script>
const allDates          = {json.dumps(dates)};
const allCumPNL         = {json.dumps(cum_pnl)};
const allDaily          = {json.dumps(daily_pnl)};
const allMLabels        = {json.dumps(monthly_labels)};
const allMVals          = {json.dumps(monthly_vals)};
const allStrategyEquity = {json.dumps(strategy_equity)};
const allNiftyEquity    = {json.dumps(nifty_equity)};
const allSensexEquity   = {json.dumps(sensex_equity)};
const allBankEquity     = {json.dumps(bank_equity)};
const allMidcapEquity   = {json.dumps(midcap_equity)};
const allSmallcapEquity = {json.dumps(smallcap_equity)};
const allIdleFund       = {json.dumps(idle_fund_vals)};
const allIdlePct        = {json.dumps(idle_pct_vals)};
const allMFNames        = {json.dumps(mf_names_list)};
const allMFEquity       = {json.dumps(mf_equities)};
const allMFDD           = {json.dumps(mf_dd_curves)};
const allStrategyDD     = {json.dumps(strategy_dd_curve)};
const allNiftyDD        = {json.dumps(nifty_dd_curve)};
const allMidcapDD       = {json.dumps(midcap_dd_curve)};
const allSmallcapDD     = {json.dumps(smallcap_dd_curve)};

const TICKS = {{ maxTicksLimit:8, maxRotation:30 }};
const GRID  = {{ color:'rgba(0,0,0,.05)' }};

/* helpers */
function reindex(arr) {{
  if (!arr || !arr.length || arr[0] === 0) return arr;
  const base = arr[0];
  return arr.map(v => parseFloat((v / base * 100).toFixed(2)));
}}

function getCutoff(months) {{
  const last = new Date(allDates[allDates.length - 1]);
  if (months === 'all') return null;
  if (months === 'ytd') return new Date(last.getFullYear(), 0, 1);
  const d = new Date(last);
  d.setMonth(d.getMonth() - parseInt(months, 10));
  return d;
}}

function sliceFrom(cutoff) {{
  if (!cutoff) return 0;
  const idx = allDates.findIndex(d => new Date(d) >= cutoff);
  return idx === -1 ? 0 : idx;
}}

function filterMonthly(cutoff) {{
  if (!cutoff) return {{ labels: allMLabels, vals: allMVals }};
  const cutStr = cutoff.getFullYear() + '-' +
                 String(cutoff.getMonth() + 1).padStart(2, '0');
  const idx = allMLabels.findIndex(l => l >= cutStr);
  if (idx === -1) return {{ labels: [], vals: [] }};
  return {{ labels: allMLabels.slice(idx), vals: allMVals.slice(idx) }};
}}

function ddFromEquity(equity) {{
  let peak = -Infinity;
  return equity.map(v => {{
    if (v > peak) peak = v;
    return parseFloat(((v / peak - 1) * 100).toFixed(2));
  }});
}}

/* chart instances */
const cChart = new Chart(document.getElementById('cChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [{{
    label: 'Total returns', borderColor: '#185FA5',
    backgroundColor: 'rgba(24,95,165,.07)',
    fill: true, borderWidth: 2, tension: .3, pointRadius: 0
  }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: TICKS, grid: {{ display: false }} }}, y: {{ grid: GRID }} }}
  }}
}});

const dChart = new Chart(document.getElementById('dChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [{{ label: 'Daily P&L', borderRadius: 2 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: TICKS, grid: {{ display: false }} }}, y: {{ grid: GRID }} }}
  }}
}});

const mChart = new Chart(document.getElementById('mChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [{{ label: 'Monthly P&L', borderRadius: 3 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ autoSkip: false, maxRotation: 30 }}, grid: {{ display: false }} }},
      y: {{ grid: GRID }}
    }}
  }}
}});

const winPieChart = new Chart(document.getElementById('winPieChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Win', 'Loss', 'Flat'],
    datasets: [{{
      data: [],
      backgroundColor: [
        'rgba(59,109,17,.85)',
        'rgba(163,45,45,.85)',
        'rgba(136,135,128,.75)'
      ],
      borderWidth: 0,
      hoverOffset: 6
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    cutout: '62%',
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

const idleChart = new Chart(document.getElementById('idleChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [
    {{
      label: 'Idle fund (Rs.)',
      borderRadius: 2,
      backgroundColor: 'rgba(255,165,0,.65)',
      yAxisID: 'yAmt'
    }},
    {{
      label: 'Idle %',
      type: 'line',
      borderColor: 'rgba(163,45,45,.85)',
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: .3,
      yAxisID: 'yPct'
    }}
  ] }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: TICKS, grid: {{ display: false }} }},
      yAmt: {{
        position: 'left',
        grid: GRID,
        title: {{ display: true, text: 'Rs. idle', font: {{ size: 11 }}, color: '#888780' }}
      }},
      yPct: {{
        position: 'right',
        grid: {{ display: false }},
        title: {{ display: true, text: '% idle', font: {{ size: 11 }}, color: '#888780' }},
        ticks: {{ callback: v => v + '%' }}
      }}
    }}
  }}
}});

/* MF palette */
const MF_COLORS = [
  '#E07B39','#9B59B6','#1ABC9C','#E74C3C',
  '#3498DB','#F39C12','#2ECC71','#95A5A6'
];

const mfDatasets = allMFNames.map((name, idx) => ({{
  label: name,
  borderColor: MF_COLORS[idx % MF_COLORS.length],
  backgroundColor: 'transparent',
  borderWidth: 1.5,
  pointRadius: 0,
  tension: .3,
  data: []
}}));

const mfChart = new Chart(document.getElementById('mfChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [
    {{ label: 'Strategy', borderColor: '#185FA5', borderWidth: 3, pointRadius: 0, tension: .3, data: [] }},
    ...mfDatasets
  ] }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }}
  }}
}});

const mfDDDatasets = allMFNames.map((name, idx) => ({{
  label: name,
  borderColor: MF_COLORS[idx % MF_COLORS.length],
  backgroundColor: 'transparent',
  borderWidth: 1.5,
  pointRadius: 0,
  tension: .3,
  data: []
}}));

const mfDDChart = new Chart(document.getElementById('mfDDChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [
    {{ label: 'Strategy', borderColor: '#185FA5', borderWidth: 3, pointRadius: 0, tension: .3, data: [] }},
    ...mfDDDatasets
  ] }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }}
  }}
}});

const benchmarkChart = new Chart(document.getElementById('benchmarkChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [
    {{ label: 'Strategy',   borderWidth: 3, pointRadius: 0 }},
    {{ label: 'Nifty',      pointRadius: 0 }},
    {{ label: 'Sensex',     pointRadius: 0 }},
    {{ label: 'Bank Nifty', pointRadius: 0 }},
    {{ label: 'Midcap',     pointRadius: 0 }},
    {{ label: 'Smallcap',   pointRadius: 0 }}
  ] }},
  options: {{ responsive: true, maintainAspectRatio: false }}
}});

const drawdownChart = new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [
    {{ label: 'Strategy', borderWidth: 3, pointRadius: 0 }},
    {{ label: 'Nifty',    pointRadius: 0 }},
    {{ label: 'Midcap',   pointRadius: 0 }},
    {{ label: 'Smallcap', pointRadius: 0 }}
  ] }},
  options: {{ responsive: true, maintainAspectRatio: false }}
}});

/* main filter function */
function applyFilter(months) {{
  const cutoff = getCutoff(months);
  const i = sliceFrom(cutoff);

  const dates  = allDates.slice(i);
  const cumPNL = allCumPNL.slice(i);
  const daily  = allDaily.slice(i);
  const {{ labels: mL, vals: mV }} = filterMonthly(cutoff);

  cChart.data.labels = dates;
  cChart.data.datasets[0].data = reindex(cumPNL);
  cChart.update('none');

  dChart.data.labels = dates;
  dChart.data.datasets[0].data = daily;
  dChart.data.datasets[0].backgroundColor =
    daily.map(v => v >= 0 ? 'rgba(59,109,17,.75)' : 'rgba(163,45,45,.75)');
  dChart.update('none');

  mChart.data.labels = mL;
  mChart.data.datasets[0].data = mV;
  mChart.data.datasets[0].backgroundColor =
    mV.map(v => v >= 0 ? 'rgba(59,109,17,.75)' : 'rgba(163,45,45,.75)');
  mChart.update('none');

  const winDays   = daily.filter(v => v >  0).length;
  const lossDays  = daily.filter(v => v <  0).length;
  const flatDays  = daily.filter(v => v === 0).length;
  const totalDays = daily.length || 1;

  winPieChart.data.datasets[0].data = [winDays, lossDays, flatDays];
  winPieChart.update('none');

  const pct = n => (n / totalDays * 100).toFixed(1) + '%';
  document.getElementById('pieLegend').innerHTML = [
    ['rgba(59,109,17,.85)',   'Win',  winDays,  pct(winDays)],
    ['rgba(163,45,45,.85)',   'Loss', lossDays, pct(lossDays)],
    ['rgba(136,135,128,.75)', 'Flat', flatDays, pct(flatDays)]
  ].map(([clr, lbl, cnt, p]) =>
    `<span style="display:flex;align-items:center;gap:5px">
       <span style="width:10px;height:10px;border-radius:2px;background:${{clr}}"></span>
       <span><strong style="color:#1a1a18">${{p}}</strong> ${{lbl}} (${{cnt}}d)</span>
     </span>`
  ).join('');

  idleChart.data.labels = dates;
  idleChart.data.datasets[0].data = allIdleFund.slice(i);
  idleChart.data.datasets[1].data = allIdlePct.slice(i);
  idleChart.update('none');

  mfChart.data.labels = dates;
  mfChart.data.datasets[0].data = reindex(allStrategyEquity.slice(i));
  allMFEquity.forEach((eq, idx) => {{
    mfChart.data.datasets[idx + 1].data = reindex(eq.slice(i));
  }});
  mfChart.update('none');

  mfDDChart.data.labels = dates;
  mfDDChart.data.datasets[0].data = ddFromEquity(reindex(allStrategyEquity.slice(i)));
  allMFEquity.forEach((eq, idx) => {{
    mfDDChart.data.datasets[idx + 1].data = ddFromEquity(reindex(eq.slice(i)));
  }});
  mfDDChart.update('none');

  const equitySeries = [
    allStrategyEquity, allNiftyEquity, allSensexEquity,
    allBankEquity, allMidcapEquity, allSmallcapEquity
  ];
  benchmarkChart.data.labels = dates;
  equitySeries.forEach((s, idx) => {{
    benchmarkChart.data.datasets[idx].data = reindex(s.slice(i));
  }});
  benchmarkChart.update('none');

  const ddSeries = [
    allStrategyEquity, allNiftyEquity, allMidcapEquity, allSmallcapEquity
  ];
  drawdownChart.data.labels = dates;
  ddSeries.forEach((s, idx) => {{
    drawdownChart.data.datasets[idx].data = ddFromEquity(reindex(s.slice(i)));
  }});
  drawdownChart.update('none');
}}

/* button wiring */
document.getElementById('filterBar').addEventListener('click', function(e) {{
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilter(btn.dataset.months);
}});

applyFilter('all');
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    filepath    = sys.argv[1] if len(sys.argv) > 1 else "portfolio_data.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "investment_dashboard.html"

    print(f"Loading: {filepath}")
    df = load_data(filepath)
    print(f"  {len(df)} rows  |  {df['date'].min().date()} -> {df['date'].max().date()}")

    metrics = calculate_metrics(df)
    print_summary(metrics)
    generate_dashboard(df, metrics, output_path)
