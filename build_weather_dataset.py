"""
build_weather_dataset.py — produces combined_preprocessed_weather.parquet,
which is `combined_preprocessed.parquet` plus 7 weather columns
(WX_TEMP, WX_RHUM, WX_PRCP, WX_WSPD, WX_PRES, WX_CLDC, WX_CODE).

Reuses build_dataset() from compare_weather.py so the preprocessing matches
the notebook + the comparison run exactly. Weather is fetched/cached via
weather_hourly.parquet (will fetch on first run).

Run:
    /Users/wahidhashem/miniforge3/envs/msds/bin/python build_weather_dataset.py
"""

from pathlib import Path

from compare_weather import build_dataset

ROOT = Path("/Users/wahidhashem/Desktop/STT811-Project")
OUT = ROOT / "combined_preprocessed_weather.parquet"


def main():
    df = build_dataset(with_weather=True)
    print(f"\nwriting {len(df):,} rows × {df.shape[1]} cols -> {OUT.name}")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"done: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
