"""
Publication-ready EDA figures for the STT 811 final report.

Produces PNG + PDF versions of:
    figures/eda_class_dist.{png,pdf}    - 3-class delay distribution
    figures/eda_delay_hist.{png,pdf}    - DEP_DELAY histogram
    figures/eda_delay_by_hour.{png,pdf} - delay rate by scheduled hour
    figures/eda_correlations.{png,pdf}  - feature correlations with DELAY_CLASS
    figures/eda_carrier.{png,pdf}       - per-carrier 2x2 panel

Run:
    python eda_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}.{{png,pdf}}")


def class_distribution(df):
    counts = df["DELAY_CLASS"].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    labels = ["On-time\n(<15 min)", "Moderate\n(15-90 min)", "Severe\n(90+ min)"]
    colors = ["#2a9d3f", "#F4A261", "#8B0000"]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of Flights")
    ax.set_title("Delay Severity Class Distribution")
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pct], fontweight="bold", padding=3)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
    )
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "eda_class_dist")


def delay_histogram(df):
    clipped = df["DEP_DELAY"].clip(-30, 180)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(clipped, bins=70, color="#3C5488", edgecolor="white", linewidth=0.4)
    ax.axvline(
        15, color="#E64B35", linestyle="--", linewidth=1.4, label="15-min FAA threshold"
    )
    ax.set_xlabel("Departure Delay (min, clipped to $[-30, 180]$)")
    ax.set_ylabel("Flight Count")
    ax.set_title("Distribution of Departure Delay")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "eda_delay_hist")


def delay_by_hour(df):
    hour_delay = df[df["DEP_HOUR"] < 24].groupby("DEP_HOUR")["DEP_DEL15"].mean() * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(
        hour_delay.index,
        hour_delay.values,
        marker="o",
        linewidth=2,
        color="#3C5488",
        markersize=5,
    )
    ax.set_xlabel("Scheduled Departure Hour (0-23)")
    ax.set_ylabel("Delay Rate (%)")
    ax.set_title("Delay Rate by Scheduled Departure Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "eda_delay_by_hour")


def correlations(df):
    numeric_cols = [
        "DEP_DELAY",
        "DEP_DEL15",
        "DISTANCE",
        "DEP_HOUR",
        "DAY_OF_WEEK",
        "MONTH",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CARRIER_DELAY",
        "PREV_FLIGHT_DELAY",
        "HAS_PREV_FLIGHT",
        "AIRPORT_TRAFFIC",
        "IS_HOLIDAY",
        "DELAY_CLASS",
    ]
    avail = [c for c in numeric_cols if c in df.columns]
    corr = df[avail].corr()
    target = (
        corr["DELAY_CLASS"]
        .drop(["DELAY_CLASS", "DEP_DELAY", "DEP_DEL15"])
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    colors = ["#00A087" if v > 0 else "#E64B35" for v in target.values]
    bars = ax.barh(
        range(len(target)),
        target.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(target)))
    ax.set_yticklabels(target.index)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", lw=0.7)
    ax.set_xlabel("Pearson Correlation with DELAY_CLASS")
    ax.set_title("Feature Correlations with Delay Severity")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, target.values):
        ax.text(
            v + (0.005 if v > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=11,
            color="#222222",
        )
    plt.tight_layout()
    save(fig, "eda_correlations")


def carrier_panel():
    cols = [
        "OP_UNIQUE_CARRIER",
        "DEP_DEL15",
        "DEP_DELAY",
        "CANCELLED",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]
    df = pd.read_parquet("combined_new.parquet", columns=cols)
    df = df[df["CANCELLED"] != 1].copy()
    names = {
        "AA": "American",
        "AS": "Alaska",
        "B6": "JetBlue",
        "DL": "Delta",
        "F9": "Frontier",
        "G4": "Allegiant",
        "HA": "Hawaiian",
        "MQ": "Envoy",
        "NK": "Spirit",
        "OH": "PSA",
        "OO": "SkyWest",
        "UA": "United",
        "WN": "Southwest",
        "YX": "Republic",
        "9E": "Endeavor",
        "YV": "Mesa",
    }
    df["carrier"] = df["OP_UNIQUE_CARRIER"].map(names).fillna(df["OP_UNIQUE_CARRIER"])

    agg = df.groupby("carrier").agg(
        n_flights=("DEP_DEL15", "size"),
        delay_rate=("DEP_DEL15", "mean"),
        avg_dep_delay=("DEP_DELAY", "mean"),
        carrier_delay=("CARRIER_DELAY", "sum"),
        weather_delay=("WEATHER_DELAY", "sum"),
        nas_delay=("NAS_DELAY", "sum"),
        security_delay=("SECURITY_DELAY", "sum"),
        late_aircraft_delay=("LATE_AIRCRAFT_DELAY", "sum"),
    )
    agg["delay_rate_pct"] = agg["delay_rate"] * 100
    reasons = [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    ]
    totals = agg[reasons].sum(axis=1).replace(0, np.nan)
    reason_pct = agg[reasons].div(totals, axis=0) * 100

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

    ax = axes[0, 0]
    d = agg["delay_rate_pct"].sort_values()
    bars = ax.barh(d.index, d.values, color="#3C5488", edgecolor="white", linewidth=0.4)
    ax.set_title(
        "Delay Rate by Carrier (% flights $\\geq$15 min late)", fontweight="bold"
    )
    ax.set_xlabel("% of flights delayed")
    for bar, v in zip(bars, d.values):
        ax.text(
            v + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            fontsize=11,
        )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    d = agg["avg_dep_delay"].sort_values()
    med = d.median()
    colors = [
        "#00A087" if v < 0 else "#3C5488" if v < med else "#E64B35" for v in d.values
    ]
    bars = ax.barh(d.index, d.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444444", lw=0.7)
    ax.set_title("Average Departure Delay (min)", fontweight="bold")
    ax.set_xlabel("Avg DEP_DELAY  (negative = early)")
    for bar, v in zip(bars, d.values):
        ax.text(
            v + (0.15 if v >= 0 else -0.15),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=11,
        )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    order = agg["delay_rate_pct"].sort_values(ascending=False).index
    plot_df = reason_pct.loc[order].copy()
    plot_df.columns = [
        "Carrier",
        "Weather",
        "NAS (Air Traffic)",
        "Security",
        "Late Aircraft",
    ]
    plot_df.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        width=0.8,
        color=["#3C5488", "#00A087", "#E64B35", "#8172B2", "#CCB974"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("Delay-Minute Breakdown by Cause (%)", fontweight="bold")
    ax.set_xlabel("% of total delay minutes")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    d = (agg["n_flights"] / 1e6).sort_values()
    bars = ax.barh(d.index, d.values, color="#4DBBD5", edgecolor="white", linewidth=0.4)
    ax.set_title("Flight Volume by Carrier", fontweight="bold")
    ax.set_xlabel("Flights (millions)")
    for bar, v in zip(bars, d.values):
        ax.text(
            v + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}M",
            va="center",
            fontsize=11,
        )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "eda_carrier")


def grid_2x2(df):
    """Combined 2x2 panel: class dist, delay hist, hour-of-day, correlations."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4))

    # --- (a) class distribution
    ax = axes[0, 0]
    counts = df["DELAY_CLASS"].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    labels = ["On-time\n(<15 min)", "Moderate\n(15-90 min)", "Severe\n(90+ min)"]
    colors = ["#2a9d3f", "#F4A261", "#8B0000"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of Flights")
    ax.set_title("(a) Delay Severity Class Distribution", fontweight="bold")
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pct], fontweight="bold", padding=3)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
    )
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # --- (b) DEP_DELAY histogram
    ax = axes[0, 1]
    clipped = df["DEP_DELAY"].clip(-30, 180)
    ax.hist(clipped, bins=70, color="#3C5488", edgecolor="white", linewidth=0.4)
    ax.axvline(
        15, color="#E64B35", linestyle="--", linewidth=1.4, label="15-min FAA threshold"
    )
    ax.set_xlabel("Departure Delay (min, clipped to $[-30, 180]$)")
    ax.set_ylabel("Flight Count")
    ax.set_title("(b) Distribution of Departure Delay", fontweight="bold")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # --- (c) delay rate by hour
    ax = axes[1, 0]
    hour_delay = df[df["DEP_HOUR"] < 24].groupby("DEP_HOUR")["DEP_DEL15"].mean() * 100
    ax.plot(
        hour_delay.index,
        hour_delay.values,
        marker="o",
        linewidth=2,
        color="#3C5488",
        markersize=5,
    )
    ax.set_xlabel("Scheduled Departure Hour (0-23)")
    ax.set_ylabel("Delay Rate (%)")
    ax.set_title("(c) Delay Rate by Scheduled Hour", fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # --- (d) feature correlations
    ax = axes[1, 1]
    numeric_cols = [
        "DEP_DELAY",
        "DEP_DEL15",
        "DISTANCE",
        "DEP_HOUR",
        "DAY_OF_WEEK",
        "MONTH",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CARRIER_DELAY",
        "PREV_FLIGHT_DELAY",
        "HAS_PREV_FLIGHT",
        "AIRPORT_TRAFFIC",
        "IS_HOLIDAY",
        "DELAY_CLASS",
    ]
    avail = [c for c in numeric_cols if c in df.columns]
    corr = df[avail].corr()
    target = (
        corr["DELAY_CLASS"]
        .drop(["DELAY_CLASS", "DEP_DELAY", "DEP_DEL15"])
        .sort_values(ascending=False)
    )
    bar_colors = ["#00A087" if v > 0 else "#E64B35" for v in target.values]
    bars = ax.barh(
        range(len(target)),
        target.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(target)))
    ax.set_yticklabels(target.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", lw=0.7)
    ax.set_xlabel("Pearson Correlation with DELAY_CLASS")
    ax.set_title("(d) Feature Correlations", fontweight="bold")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, target.values):
        ax.text(
            v + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )

    plt.tight_layout()
    save(fig, "eda_grid")


def main():
    cols = [
        "DEP_DELAY",
        "DEP_DEL15",
        "DISTANCE",
        "DEP_HOUR",
        "DAY_OF_WEEK",
        "MONTH",
        "DELAY_CLASS",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CARRIER_DELAY",
        "PREV_FLIGHT_DELAY",
        "HAS_PREV_FLIGHT",
        "AIRPORT_TRAFFIC",
        "IS_HOLIDAY",
    ]
    df = pd.read_parquet("combined_preprocessed.parquet", columns=cols)
    print(f"Loaded {len(df):,} rows")

    class_distribution(df)
    delay_histogram(df)
    delay_by_hour(df)
    correlations(df)
    grid_2x2(df)
    carrier_panel()
    print(f"\nDone. Figures in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
