from pathlib import Path

from compare_weather import build_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "combined_preprocessed_weather.parquet"


def main():
    df = build_dataset(with_weather=True)
    print(f"\nwriting {len(df):,} rows × {df.shape[1]} cols -> {OUT.name}")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"done: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
