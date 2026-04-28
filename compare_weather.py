"""
compare_weather.py — replicates the notebook pipeline EXACTLY and runs every
model twice: once without weather, once with weather. Prints macro F1 deltas
so we can decide if weather is worth adding to the notebook.

Pipeline mirrors:
  - data_preprocess.ipynb  (combined_new.parquet -> preprocessed features)
  - model_analysis.ipynb   (ColumnTransformer + 500k sample + 4 models)

Models: Logistic Regression, Random Forest, XGBoost, Mixed Naive Bayes.
RNN is skipped here because tensorflow lives in a different conda env.

Run from the msds env:
    /Users/wahidhashem/miniforge3/envs/msds/bin/python compare_weather.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/wahidhashem/Desktop/STT811-Project")
WEATHER_CACHE = ROOT / "weather_hourly.parquet"

SEED = 42
SAMPLE_SIZE = 500_000
PER_CLASS = 100_000


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def fetch_weather() -> pd.DataFrame:
    """Hourly weather for the top-80 origin airports in 2025 (Meteostat)."""
    if WEATHER_CACHE.exists():
        print(f"  using cached weather: {WEATHER_CACHE.name}")
        return pd.read_parquet(WEATHER_CACHE)

    print("  weather cache missing — fetching from Meteostat (3-6 min) ...")
    import airportsdata
    from meteostat import Point, hourly, stations

    airports = airportsdata.load("IATA")
    top = (
        pd.read_parquet(ROOT / "combined_new.parquet", columns=["ORIGIN"])
        ["ORIGIN"].value_counts().head(80).index.tolist()
    )

    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31, 23, 59)

    frames = []
    for i, iata in enumerate(top, 1):
        a = airports.get(iata)
        if a is None:
            print(f"    [{i}/80] {iata}: no coords, skip")
            continue
        try:
            near = stations.nearby(Point(a["lat"], a["lon"], a["elevation"])).head(1)
            if near.empty:
                print(f"    [{i}/80] {iata}: no nearby station")
                continue
            station_id = near.index[0]
            w = hourly(station_id, start, end).fetch()
        except Exception as e:
            print(f"    [{i}/80] {iata}: fetch error {e}")
            continue
        if w is None or w.empty:
            print(f"    [{i}/80] {iata}: empty")
            continue
        w = w.reset_index()
        # Meteostat v2 can return Parameter enums as column labels — stringify
        w.columns = [str(c).split(".")[-1].lower().rstrip("'>") for c in w.columns]
        w["ORIGIN"] = iata
        frames.append(w)
        print(f"    [{i}/80] {iata} ({station_id}): {len(w):,} rows")

    wx = pd.concat(frames, ignore_index=True)
    # Meteostat hourly fields:
    #   WX_TEMP — air temperature (°C)
    #   WX_RHUM — relative humidity (%)
    #   WX_PRCP — precipitation total over the hour (mm)
    #   WX_WSPD — average wind speed (km/h)
    #   WX_PRES — sea-level air pressure (hPa)
    #   WX_CLDC — cloud cover (oktas, 0=clear … 8=overcast)
    #   WX_CODE — weather condition code (1=clear, 5=fog, 9=rain,
    #             18=thunderstorm, etc — full list in Meteostat docs)
    wx = wx.rename(columns={
        "time": "WX_TIME", "temp": "WX_TEMP", "rhum": "WX_RHUM",
        "prcp": "WX_PRCP", "wspd": "WX_WSPD", "pres": "WX_PRES",
        "cldc": "WX_CLDC", "coco": "WX_CODE",
    })
    keep = ["ORIGIN", "WX_TIME", "WX_TEMP", "WX_RHUM",
            "WX_PRCP", "WX_WSPD", "WX_PRES", "WX_CLDC", "WX_CODE"]
    wx = wx[[c for c in keep if c in wx.columns]]
    wx.to_parquet(WEATHER_CACHE, index=False, compression="zstd")
    print(f"  saved {len(wx):,} rows -> {WEATHER_CACHE.name}")
    return wx


# ---------------------------------------------------------------------------
# Data build — mirrors data_preprocess.ipynb cell-by-cell
# ---------------------------------------------------------------------------

def build_dataset(with_weather: bool) -> pd.DataFrame:
    label = "WITH WEATHER" if with_weather else "NO WEATHER"
    print(f"\n[{label}] building dataset from combined_new.parquet ...")

    df = pd.read_parquet(ROOT / "combined_new.parquet")
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df = df.drop_duplicates()
    df = df[df["CANCELLED"] == 0].copy()
    df = df.dropna(subset=["DEP_DELAY"])

    # 5-class target
    def bin_delay(m):
        if m < 15:  return 0
        if m < 30:  return 1
        if m < 60:  return 2
        if m < 120: return 3
        return 4
    df["DELAY_CLASS"] = df["DEP_DELAY"].apply(bin_delay)

    # State + dep hour
    df["ORIGIN_STATE"] = df["ORIGIN_CITY_NAME"].str.split(",").str[-1].str.strip()
    df["DEST_STATE"]   = df["DEST_CITY_NAME"].str.split(",").str[-1].str.strip()
    df["DEP_HOUR"]     = df["CRS_DEP_TIME"] // 100

    # PREV_FLIGHT_DELAY via TAIL_NUM + same-day shift
    df = df.sort_values(["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"]).reset_index(drop=True)
    df["_PREV_ARR"]  = df.groupby("TAIL_NUM")["ARR_DELAY"].shift(1)
    df["_PREV_DATE"] = df.groupby("TAIL_NUM")["FL_DATE"].shift(1)
    same_day = df["_PREV_DATE"] == df["FL_DATE"]
    df["PREV_FLIGHT_DELAY"] = np.where(same_day, df["_PREV_ARR"], np.nan)
    df["HAS_PREV_FLIGHT"]   = same_day.astype(int)
    df["PREV_FLIGHT_DELAY"] = df["PREV_FLIGHT_DELAY"].fillna(0)
    df = df.drop(columns=["_PREV_ARR", "_PREV_DATE"])

    if with_weather:
        wx = fetch_weather().copy()
        wx["WX_TIME"] = pd.to_datetime(wx["WX_TIME"])
        wx["WX_DATE"] = wx["WX_TIME"].dt.normalize()
        wx["WX_HOUR"] = wx["WX_TIME"].dt.hour
        wx = wx.drop(columns=["WX_TIME"]).drop_duplicates(
            subset=["ORIGIN", "WX_DATE", "WX_HOUR"]
        )
        df["_HOUR"] = df["DEP_HOUR"].replace(24, 0)
        before = len(df)
        df = df.merge(
            wx, how="left",
            left_on=["ORIGIN", "FL_DATE", "_HOUR"],
            right_on=["ORIGIN", "WX_DATE", "WX_HOUR"],
        )
        df = df.drop(columns=["_HOUR", "WX_DATE", "WX_HOUR"])
        assert len(df) == before, f"merge changed row count {before} -> {len(df)}"
        wx_cols = [c for c in df.columns if c.startswith("WX_")]
        coverage = df[wx_cols[0]].notna().mean()
        df[wx_cols] = df[wx_cols].fillna(0)  # missing -> 0 (airport not in top 80)
        print(f"  merged {len(wx_cols)} weather cols, coverage={coverage:.1%}")

    drop_cols = [
        "ARR_DELAY", "DEP_DELAY_NEW", "CANCELLED", "FL_DATE", "CRS_DEP_TIME",
        "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "TAIL_NUM", "ORIGIN", "DEST",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  final shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Models — match the notebook's setup exactly
# ---------------------------------------------------------------------------

def run_main_models(df: pd.DataFrame, label: str) -> dict:
    """LR, RF, XGBoost with shared ColumnTransformer + 500k sample."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    import xgboost as xgb

    leak = ["DEP_DELAY", "DEP_DEL15", "CARRIER_DELAY", "WEATHER_DELAY",
            "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
    X = df.drop(columns=[c for c in leak + ["DELAY_CLASS"] if c in df.columns])
    X["DEP_HOUR"] = X["DEP_HOUR"].replace(24, 0)
    y = df["DELAY_CLASS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    num_cols = X_train.select_dtypes(
        include=["int8", "int16", "int32", "int64",
                 "float16", "float32", "float64"]
    ).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    pre = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    X_tr_enc = pre.fit_transform(X_train)
    X_te_enc = pre.transform(X_test)
    print(f"  [{label}] encoded: train={X_tr_enc.shape}, test={X_te_enc.shape}")

    rng = np.random.RandomState(SEED)
    idx = rng.choice(X_tr_enc.shape[0], SAMPLE_SIZE, replace=False)
    X_tr = X_tr_enc[idx]
    y_tr = y_train.iloc[idx]

    def score(name, y_pred):
        mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        print(f"    {name:<22} macro F1={mf1:.4f}  acc={acc:.4f}")
        return {"macro_f1": mf1, "accuracy": acc}

    out = {}

    print(f"  [{label}] fitting Logistic Regression ...")
    lr = LogisticRegression(
        solver="saga", max_iter=200, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    lr.fit(X_tr, y_tr)
    out["Logistic Regression"] = score("Logistic Regression", lr.predict(X_te_enc))

    print(f"  [{label}] fitting Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    out["Random Forest"] = score("Random Forest", rf.predict(X_te_enc))

    print(f"  [{label}] fitting XGBoost ...")
    counts = np.bincount(y_tr.values)
    w_class = len(y_tr) / (len(counts) * counts)
    w_sample = w_class[y_tr.values]
    xgm = xgb.XGBClassifier(
        objective="multi:softprob", num_class=5, max_depth=5,
        learning_rate=0.2, n_estimators=100, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        n_jobs=-1, tree_method="hist",
    )
    xgm.fit(X_tr, y_tr.values, sample_weight=w_sample, verbose=False)
    out["XGBoost"] = score("XGBoost", xgm.predict(X_te_enc))

    return out


def run_nb(df: pd.DataFrame, label: str) -> dict:
    """Mixed Naive Bayes — own preprocessing (ordinal encode + balanced sample)."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from mixed_naive_bayes import MixedNB

    leak = ["DEP_DELAY", "ARR_DELAY", "DEP_DEL15", "CARRIER_DELAY",
            "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY", "DIVERTED", "DEST_AIRPORT_SEQ_ID"]
    X_nb = df.drop(columns=[c for c in leak + ["DELAY_CLASS"] if c in df.columns])
    X_nb["DEP_HOUR"] = X_nb["DEP_HOUR"].replace(24, 0)
    y_nb = df["DELAY_CLASS"]

    NUMERIC_COLS = {"DISTANCE", "CRS_ELAPSED_TIME", "PREV_FLIGHT_DELAY"}
    NUMERIC_COLS.update({c for c in X_nb.columns if c.startswith("WX_")})

    X_train, X_test, y_train, y_test = train_test_split(
        X_nb, y_nb, test_size=0.2, random_state=SEED, stratify=y_nb,
    )

    cat_cols = [c for c in X_nb.columns if c not in NUMERIC_COLS]
    enc = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X_nb[cat_cols])

    min_count = min(y_train.value_counts().min(), PER_CLASS)
    X_tr_bal = (
        X_train.groupby(y_train, group_keys=False)
        .apply(lambda g: g.sample(min_count, random_state=SEED))
        .reset_index(drop=True)
    )
    y_tr_bal = (
        y_train.groupby(y_train, group_keys=False)
        .apply(lambda g: g.sample(min_count, random_state=SEED))
        .reset_index(drop=True)
    )

    def pipe(data):
        data = data.copy()
        cat_idx = [i for i, c in enumerate(data.columns) if c not in NUMERIC_COLS]
        data.iloc[:, cat_idx] = enc.transform(data.iloc[:, cat_idx])
        return data.astype(float)

    X_tr_bal = pipe(X_tr_bal)
    X_test_enc = pipe(X_test)

    cat_idx = [i for i, c in enumerate(X_tr_bal.columns) if c not in NUMERIC_COLS]
    print(f"  [{label}] fitting MixedNB ({len(X_tr_bal):,} rows, {min_count:,}/class) ...")
    m = MixedNB(categorical_features=cat_idx)
    m.fit(X_tr_bal, y_tr_bal)
    p = m.predict(X_test_enc)
    mf1 = f1_score(y_test, p, average="macro", zero_division=0)
    acc = accuracy_score(y_test, p)
    print(f"    {'Mixed Naive Bayes':<22} macro F1={mf1:.4f}  acc={acc:.4f}")
    return {"Mixed Naive Bayes": {"macro_f1": mf1, "accuracy": acc}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(" Weather-vs-no-weather comparison (pipeline mirrors the notebooks)")
    print(f"  seed={SEED}   sample={SAMPLE_SIZE:,}   NB balanced={PER_CLASS:,}/class")
    print("=" * 78)

    results = {}
    for with_wx, label in [(False, "NO_WEATHER"), (True, "WITH_WEATHER")]:
        df = build_dataset(with_wx)
        res = run_main_models(df, label)
        res.update(run_nb(df, label))
        results[label] = res
        del df

    # Summary
    models = list(results["NO_WEATHER"].keys())
    print("\n" + "=" * 78)
    print(f"{'Model':<22}  {'No wx F1':>10}  {'+wx F1':>8}  {'Δ macro F1':>11}  {'verdict':>12}")
    print("-" * 78)
    deltas = []
    rows = []
    for name in models:
        a = results["NO_WEATHER"][name]["macro_f1"]
        b = results["WITH_WEATHER"][name]["macro_f1"]
        d = b - a
        deltas.append(d)
        verdict = "helps" if d > 0.005 else ("hurts" if d < -0.005 else "no change")
        print(f"{name:<22}  {a:>10.4f}  {b:>8.4f}  {d:>+11.4f}  {verdict:>12}")
        rows.append({
            "model": name,
            "macro_f1_no_wx": a, "macro_f1_wx": b, "delta": d,
            "acc_no_wx": results["NO_WEATHER"][name]["accuracy"],
            "acc_wx":    results["WITH_WEATHER"][name]["accuracy"],
        })
    avg = float(np.mean(deltas))
    print("-" * 78)
    print(f"{'AVERAGE Δ':<22}  {'':>10}  {'':>8}  {avg:>+11.4f}")
    print("=" * 78)
    if avg > 0.005:
        print(" VERDICT: KEEP weather — consistent improvement across models.")
    elif avg < -0.005:
        print(" VERDICT: DROP weather — actively hurts average performance.")
    else:
        print(" VERDICT: SKIP weather — no measurable benefit, not worth the complexity.")
    print("=" * 78)

    out_csv = ROOT / "weather_comparison_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f" results saved -> {out_csv.name}")


if __name__ == "__main__":
    sys.exit(main())
