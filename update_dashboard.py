# -*- coding: utf-8 -*-
"""
CALENDAR_REGIME_DASHBOARD - updater diario (patron COCKPIT Batman LT).
Genera data.json con: GATE (Z50>=1.31 AND BBW>=175, la DECISION binaria), dials informativos
(Z50, BBW, RADAR viento, SMA7p) y CALENDAR_COCKPIT score V2 (2026-07-14, estudio zigzag:
score = pctE(Z50) con veto VIX, SIN SMA7p). El V1 mean(pctE Z50, pctE SMA7p) tenia zigzag
ilogico: 88% de los saltos >25pts los causaba el percentil de SMA7p (indicador de momentum
rapido, no de regimen); bootstrap PAREADO V1 vs V2 = EMPATE ns en dinero, y el canon ya
decia Z50 SUBSUME SMA7. V2: r_day +0.267, LOYO 5/6 (caveat 2022 -0.38), bootL50
[+0.128,+0.379] SIG, drop-top3 MANTIENE; 60% menos saltos grandes. Ver
Calendar/ANALISIS/APR_CALENDAR_COCKPIT_20260714/estudio_zigzag_simplicidad.py.
Cohortes historicas embebidas (congeladas, con fecha de calibracion).
Publica a GitHub Pages (repo CALENDAR_REGIME). rc: 0 OK / 2 NO-DATA-WARN / 3 IDEMPOTENT / >=10 FAIL.
Fuentes (read-only salvo el feed VIX propio):
  - SPX closes: C:/Users/Administrator/Desktop/FINAL DATA/SP_SPX_CLOSE_HISTORICAL_PRICES.csv (Step 9)
  - VIX: feeds/VIX_DAILY.csv (seed FINAL DATA/VIX_CLOSE_HISTORICAL_PRICES.csv + append yfinance ^VIX)
  - RADAR: Calendar/ANALISIS/BURST_RADAR_20260610/BURST_RADAR_LITE_DAILY.csv (Step 8c)
  - Cross-check: ESTRATEGIAS/SPX_REGIME/data/SPX_REGIME_LATEST.json (Step 9)
Canon: gate = decision ON/OFF; el gradiente Z50 NO es senal de tamano; RADAR = viento IV, no dinero.
ASCII-only.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPX_CSV = Path(r"C:/Users/Administrator/Desktop/FINAL DATA/SP_SPX_CLOSE_HISTORICAL_PRICES.csv")
VIX_SEED = Path(r"C:/Users/Administrator/Desktop/FINAL DATA/VIX_CLOSE_HISTORICAL_PRICES.csv")
VIX_FEED = HERE / "feeds" / "VIX_DAILY.csv"
RADAR_CSV = Path(r"C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/Calendar/ANALISIS/BURST_RADAR_20260610/BURST_RADAR_LITE_DAILY.csv")
LATEST_JSON = Path(r"C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/SPX_REGIME/data/SPX_REGIME_LATEST.json")
DATA_JSON = HERE / "data.json"

SERIES_START = "2019-01-01"   # era de calibracion del APR (consistencia de percentiles)
GATE_Z50 = 1.31
GATE_Z50_CONV = 1.60
GATE_BBW = 175.0
PCTE_MIN = 250                # min_periods del percentil expanding (APR)
VIX_VETO_PCT = 0.67
ZONE_ROJA, ZONE_VERDE = 20.0, 80.0
STALE_WARN = {"Z50": 2, "BBW": 2, "RADAR": 2, "SMA7P": 2, "VIX": 3}
STALE_SCORE_MAX = 5
PUSH_ENABLED = True

# ---- Cohortes historicas EMBEBIDAS (congeladas; ver docs de trazabilidad) ----
COHORTS = {
    "gate_bands": {  # madre day-level, mean %/dia y PF_dia (auditoria masiva Z50 A4, 2026-07-07)
        "calibrated": "2026-07-07 (madre 253,733 tr / 1,735 dias)",
        "rows": [
            {"band": "Z50 < 0",        "dias": 463, "mean_day": -3.52, "pf_day": 0.71},
            {"band": "0 - 1.31",       "dias": 601, "mean_day": 0.16,  "pf_day": 1.02},
            {"band": "1.31 - 1.5",     "dias": 171, "mean_day": 6.35,  "pf_day": 2.39},
            {"band": "1.5 - 2.0",      "dias": 413, "mean_day": 9.60,  "pf_day": 3.38},
            {"band": ">= 2.0",         "dias": 87,  "mean_day": 10.65, "pf_day": 3.64},
        ],
        "note": ("Referencia historica pooled. El premium fino de Z50 alto NO es senal de "
                 "tamano (episodios; auditoria 2026-07-07). Gate = ON/OFF en 1.31."),
    },
    "cockpit_zones": {  # COCKPIT V2 = pctE(Z50)+veto (estudio zigzag 2026-07-14), madre 1,485 dias
        "calibrated": "2026-07-14 V2 Z50-only (madre 1,485 dias; LOYO 5/6, caveat 2022 -0.38)",
        "rows": [
            {"zone": "ROJA (<=20)",   "dias": 316, "mean_day": -4.83, "pf_day": 0.66, "cvar5": -77.2},
            {"zone": "NEUTRA",        "dias": 916, "mean_day": 2.97,  "pf_day": 1.46, "cvar5": -55.7},
            {"zone": "VERDE (>=80)",  "dias": 253, "mean_day": 11.22, "pf_day": 3.73, "cvar5": -56.3},
        ],
        "note": ("Premium VERDE sobrevive drop-top-3 episodios (+10.25 -> +7.41). V2 sin SMA7p: "
                 "dinero EMPATE vs V1 (boot pareado ns) con 60% menos zigzag."),
    },
    "sma7p_terciles_gate": {  # estudio EXANTE_MONEY_STACK 2026-07-13 (dentro del gate)
        "calibrated": "2026-07-13 (gated 671 dias)",
        "rows": [
            {"tercil": "T1 bajo", "dias": 224, "mean_day": 5.45,  "pf_day": 2.02},
            {"tercil": "T2",      "dias": 223, "mean_day": 9.99,  "pf_day": 3.54},
            {"tercil": "T3 alto", "dias": 224, "mean_day": 11.26, "pf_day": 4.49},
        ],
        "note": "Dial informativo (LOYO 5/6; 2021 fallo). NO es gate ni senal de tamano.",
    },
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def busdays_since(d):
    try:
        return int(np.busday_count(np.datetime64(pd.Timestamp(d).date()), np.datetime64(datetime.now().date())))
    except Exception:
        return 99

def pct_expanding(values, min_periods=PCTE_MIN):
    """Percentil expanding asof (incluye historia previa, searchsorted right)."""
    import bisect
    out = np.full(len(values), np.nan)
    ref = []
    for i, x in enumerate(values):
        if np.isfinite(x):
            if len(ref) >= min_periods:
                out[i] = bisect.bisect_right(ref, x) / len(ref)
            bisect.insort(ref, x)
    return out

def load_spx_daily():
    df = pd.read_csv(SPX_CSV, usecols=["time", "close"], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    c = df["close"]
    df["sma50"] = c.rolling(50).mean(); df["std50"] = c.rolling(50).std(ddof=1)
    df["z50"] = (c - df["sma50"]) / df["std50"]
    df["sma20"] = c.rolling(20).mean(); df["std20"] = c.rolling(20).std(ddof=1)
    df["bbw"] = 4.0 * df["std20"]                     # Upper-Lower Bollinger(20,2) = 4*std20 (V23)
    df["sma7"] = c.rolling(7).mean()
    df["sma7p"] = 100.0 * (c - df["sma7"]) / df["sma7"]
    return df

def load_vix_daily():
    VIX_FEED.parent.mkdir(parents=True, exist_ok=True)
    if VIX_FEED.exists():
        v = pd.read_csv(VIX_FEED)
    else:
        seed = pd.read_csv(VIX_SEED, usecols=["time", "close"], low_memory=False)
        v = seed.rename(columns={"close": "vix"})[["time", "vix"]]
    v["time"] = pd.to_datetime(v["time"], errors="coerce")
    v["vix"] = pd.to_numeric(v["vix"], errors="coerce")
    v = v.dropna().drop_duplicates(subset=["time"], keep="last").sort_values("time")
    # append dias faltantes via yfinance ^VIX
    try:
        import yfinance as yf
        last = v["time"].max()
        if (datetime.now() - last.to_pydatetime()).days >= 1:
            fresh = yf.download("^VIX", start=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                progress=False, auto_adjust=False)
            if fresh is not None and len(fresh):
                fresh = fresh.reset_index()
                closecol = "Close" if "Close" in fresh.columns else fresh.columns[4]
                add = pd.DataFrame({"time": pd.to_datetime(fresh["Date"]),
                                    "vix": pd.to_numeric(fresh[closecol].squeeze(), errors="coerce")}).dropna()
                v = pd.concat([v, add[~add["time"].isin(v["time"])]], ignore_index=True).sort_values("time")
                log(f"VIX feed: +{len(add)} filas yfinance (hasta {v['time'].max().date()})")
    except Exception as e:
        log(f"WARN yfinance ^VIX: {type(e).__name__}: {e} (sigo con historico)")
    tmp = VIX_FEED.with_suffix(".tmp")
    v.to_csv(tmp, index=False); tmp.replace(VIX_FEED)
    return v.reset_index(drop=True)

def git_push():
    try:
        for cmd in (["git", "add", "-A"],
                    ["git", "commit", "-m", f"daily refresh {datetime.now().strftime('%Y-%m-%d')}"],
                    ["git", "push", "origin", "main"]):
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=120)
            if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
                log(f"git {' '.join(cmd[1:])}: rc={r.returncode} {(r.stderr or r.stdout).strip()[:160]}")
    except Exception as e:
        log(f"WARN git push: {type(e).__name__}: {e}")

def main():
    t0 = time.time()
    if not SPX_CSV.exists():
        log(f"FAIL: no existe {SPX_CSV}"); return 10

    spx = load_spx_daily()
    vix = load_vix_daily()
    rad = pd.read_csv(RADAR_CSV)
    rad["date"] = pd.to_datetime(rad["date"], errors="coerce")
    rad = rad.dropna(subset=["date", "RADAR_LITE"]).sort_values("date")

    # merge diario (era APR: >= SERIES_START para percentiles consistentes con la calibracion)
    d = spx.merge(vix.rename(columns={"time": "time"}), on="time", how="left")
    d = d.merge(rad[["date", "RADAR_LITE", "SEMAFORO"]].rename(columns={"date": "time"}), on="time", how="left")
    d = d[d["time"] >= pd.Timestamp(SERIES_START)].reset_index(drop=True)
    if len(d) < 300:
        log("NO-DATA-WARN: serie diaria demasiado corta"); return 2

    d["pz"] = pct_expanding(d["z50"].values)
    d["p7"] = pct_expanding(d["sma7p"].values)   # solo dial informativo (fuera del score desde V2)
    d["pv"] = pct_expanding(d["vix"].values)
    # COCKPIT V2 (2026-07-14): score = pctE(Z50) puro. SMA7p EXCLUIDO del score (estudio
    # zigzag: causaba el 88% de los saltos >25pts sin aportar dinero; boot pareado EMPATE).
    score = d["pz"] * 100.0
    veto = d["pv"] >= VIX_VETO_PCT
    d["score"] = np.where(veto & np.isfinite(score), np.minimum(score, 50.0), score)
    d["zone"] = np.select([d["score"] <= ZONE_ROJA, d["score"] >= ZONE_VERDE], ["ROJA", "VERDE"], default="NEUTRA")
    d.loc[~np.isfinite(d["score"]), "zone"] = "SIN_DATO"

    last = d.iloc[-1]
    last_date = last["time"].strftime("%Y-%m-%d")

    # idempotencia
    if DATA_JSON.exists():
        try:
            prev = json.loads(DATA_JSON.read_text(encoding="utf-8"))
            if prev.get("latest", {}).get("date") == last_date:
                log(f"IDEMPOTENT: {last_date} ya publicado"); return 3
        except Exception:
            pass

    # dials + frescura (cada fuente con su ultima fecha valida)
    def dial(name, series_time, value, pct=None, extra=None):
        dt_last = series_time.max()
        stale = busdays_since(dt_last)
        out = {"date": pd.Timestamp(dt_last).strftime("%Y-%m-%d"),
               "value": None if value is None or not np.isfinite(value) else round(float(value), 4),
               "stale_busdays": stale, "warn": stale > STALE_WARN.get(name, 2)}
        if pct is not None and np.isfinite(pct): out["pct"] = round(float(pct) * 100, 1)
        if extra: out.update(extra)
        return out

    rad_last = rad.iloc[-1] if len(rad) else None
    dials = {
        "Z50":   dial("Z50", spx["time"], last["z50"], last["pz"]),
        "BBW":   dial("BBW", spx["time"], last["bbw"], extra={"pass": bool(last["bbw"] >= GATE_BBW)}),
        "SMA7P": dial("SMA7P", spx["time"], last["sma7p"], last["p7"]),
        "RADAR": dial("RADAR", rad["date"] if len(rad) else spx["time"],
                      rad_last["RADAR_LITE"] if rad_last is not None else None,
                      extra={"semaforo": str(rad_last["SEMAFORO"]) if rad_last is not None else "N/A",
                             "label": "viento IV - informativo, NO dinero"}),
        "VIX":   dial("VIX", vix["time"], last["vix"], last["pv"]),
    }
    core_stale = max(dials[k]["stale_busdays"] for k in ("Z50", "BBW", "SMA7P"))
    indeterminado = core_stale > STALE_SCORE_MAX

    z, b = float(last["z50"]), float(last["bbw"])
    gate_status = "CERRADO" if z < GATE_Z50 else ("ABIERTO" if z < GATE_Z50_CONV else "CONVICCION")
    operable = (z >= GATE_Z50) and (b >= GATE_BBW)
    cockpit_zone = "INDETERMINADO" if indeterminado else str(last["zone"])

    # cross-check con SPX_REGIME_LATEST.json (aviso, no fatal)
    try:
        lat = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        if lat.get("dia") == last_date and abs(float(lat.get("Z50", 9e9)) - z) > 0.02:
            log(f"WARN cross-check Z50: dashboard {z:+.4f} vs Step9 {lat.get('Z50')}")
    except Exception:
        pass

    n_tail = len(d)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "formula": ("GATE = Z50>=1.31 AND BBW>=175 (decision ON/OFF). COCKPIT_CAL V2 = pctE(Z50) "
                    "con veto VIX (pctE>=0.67 -> cap 50). Zonas ROJA<=20/VERDE>=80. SMA7p = solo dial."),
        "apr": ("V2 Z50-only certificado 2026-07-14 (estudio zigzag: r_day +0.267, LOYO 5/6, bootL50 "
                "[+0.128,+0.379] SIG, drop-top3 MANTIENE; vs V1 EMPATE ns en dinero con 60% menos "
                "zigzag; caveat 2022 -0.38)"),
        "n_days": int(np.isfinite(d["score"]).sum()),
        "thresholds": {"gate_z50": GATE_Z50, "gate_z50_conviccion": GATE_Z50_CONV, "gate_bbw": GATE_BBW,
                       "roja_max": ZONE_ROJA, "verde_min": ZONE_VERDE, "vix_veto_pct": VIX_VETO_PCT * 100},
        "latest": {
            "date": last_date,
            "gate": {"status": gate_status, "operable": "SI" if operable else "NO",
                     "z50": round(z, 4), "z50_pct": dials["Z50"].get("pct"),
                     "bbw": round(b, 1), "bbw_pass": bool(b >= GATE_BBW)},
            "dials": dials,
            "cockpit": {"score_pct": None if not np.isfinite(last["score"]) else round(float(last["score"]), 1),
                        "zone": cockpit_zone,
                        "veto_vix": bool(veto.iloc[-1]) if np.isfinite(last["pv"]) else False,
                        "reason": "dial core stale >5bd" if indeterminado else ""},
        },
        "series": {
            "dates": d["time"].dt.strftime("%Y-%m-%d").tolist(),
            "score_pct": [None if not np.isfinite(x) else round(float(x), 2) for x in d["score"]],
            "z50": [None if not np.isfinite(x) else round(float(x), 3) for x in d["z50"]],
            "sma7p": [None if not np.isfinite(x) else round(float(x), 3) for x in d["sma7p"]],
            "radar": [None if not np.isfinite(x) else round(float(x), 1) for x in d["RADAR_LITE"]],
            "spx": [None if not np.isfinite(x) else round(float(x), 2) for x in d["close"]],
        },
        "cohorts": COHORTS,
        "notes": {
            "anti_sizing": ("El gate es ON/OFF. El premium de Z50/score alto se apoya en pocos episodios: "
                            "NO dimensionar por decil/zona (auditorias 2026-07-07 / 2026-07-14)."),
            "radar_wind": "BURST_RADAR y CAL_BURST leen VIENTO IV (expectativas del path), NO dinero.",
        },
    }
    tmp = DATA_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(DATA_JSON)
    log(f"data.json escrito: {last_date} | gate={gate_status}/{'SI' if operable else 'NO'} "
        f"| score={payload['latest']['cockpit']['score_pct']} ({cockpit_zone}) | n_days={payload['n_days']} "
        f"| {time.time()-t0:.1f}s")

    if PUSH_ENABLED and (HERE / ".git").exists():
        git_push()
    return 0

if __name__ == "__main__":
    sys.exit(main())
