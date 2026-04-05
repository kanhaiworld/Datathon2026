"""
Uses the PELT (Pruned Exact Linear Time) changepoint detection algorithm to
identify whether each restaurant in a Yelp reviews CSV has ever "blown up" —
i.e. experienced a sustained, structural surge in review volume.

Eligibility for PELT analysis:
  • ≥ 30 reviews total
  • ≥ 12 months between first and last review

For ineligible restaurants a result is still written but with
  blown_up = False  and  blowup_reason = "insufficient_data"

Output columns (one row per business_id):
  business_id             - original id
  review_count            - total reviews in dataset
  months_active           - months between first and last review
  eligible                - whether PELT was run (bool)
  blown_up                - whether a blowup was detected (bool)
  blowup_reason           - human-readable explanation or "none"
  n_changepoints          - number of structural breaks found by PELT
  baseline_rate           - mean reviews/month in the first segment
  peak_rate               - mean reviews/month in the peak segment
  blowup_magnitude        - peak_rate / baseline_rate  (1.0 if no blowup)
  blowup_month            - calendar month when peak segment began (or "")
  post_peak_sustained     - True if elevated rate persisted ≥ 3 months
  penalty_used            - PELT penalty value used (for transparency)
"""

import pandas as pd
import numpy as np
import ruptures as rpt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH  = "ca_restaurant_reviews_with_sentiment_combined.csv"
OUTPUT_PATH = "detected_blowups.csv"

MIN_REVIEWS = 30      # minimum reviews to run PELT
MIN_MONTHS = 12       # minimum active months to run PELT
BLOWUP_RATIO = 3.0    # peak segment must be ≥ 3× baseline to count
SUSTAINED_MONTHS = 3  # peak segment must last ≥ this many months
PELT_PENALTY = 10     # PELT regularization — higher = fewer breakpoints
PELT_MIN_SIZE = 3     # minimum segment length in months

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading reviews…")
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
df["month"] = df["date"].dt.to_period("M")

total_businesses = df["business_id"].nunique()
print(f"  {len(df):,} reviews · {total_businesses:,} businesses")

# ── Per-business analysis ─────────────────────────────────────────────────────

def analyze_business(biz_id: str, grp: pd.DataFrame) -> dict:
    base = {
        "business_id": biz_id,
        "review_count": len(grp),
        "months_active": 0,
        "eligible": False,
        "blown_up": False,
        "blowup_reason": "none",
        "n_changepoints": 0,
        "baseline_rate": 0.0,
        "peak_rate": 0.0,
        "blowup_magnitude": 1.0,
        "blowup_month": "",
        "post_peak_sustained": False,
        "penalty_used": PELT_PENALTY,
    }

    # ── Eligibility checks ────────────────────────────────────────────────────
    first, last = grp["date"].min(), grp["date"].max()
    months_active = (last - first).days / 30
    base["months_active"] = round(months_active, 1)

    if len(grp) < MIN_REVIEWS or months_active < MIN_MONTHS:
        base["blowup_reason"] = "insufficient_data"
        return base

    base["eligible"] = True

    # ── Build contiguous monthly time series ──────────────────────────────────
    monthly = grp.groupby("month").size()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range, fill_value=0)
    signal = monthly.values.astype(float)

    # Need at least PELT_MIN_SIZE * 2 months for a meaningful split
    if len(signal) < PELT_MIN_SIZE * 2:
        base["blowup_reason"] = "insufficient_data"
        return base

    # ── Run PELT ──────────────────────────────────────────────────────────────
    try:
        model = rpt.Pelt(model="rbf", min_size=PELT_MIN_SIZE, jump=1)
        model.fit(signal.reshape(-1, 1))
        breakpoints = model.predict(pen=PELT_PENALTY)
        # ruptures always appends len(signal) as the final breakpoint
        # breakpoints = [..., len(signal)]
    except Exception as e:
        base["blowup_reason"] = f"pelt_error: {e}"
        return base

    # Number of true interior changepoints (exclude the terminal one)
    n_cp = len(breakpoints) - 1
    base["n_changepoints"] = n_cp

    # With 0 changepoints there's only one segment — no blowup possible
    if n_cp == 0:
        base["blowup_reason"] = "no_changepoints_detected"
        return base

    # ── Segment analysis ──────────────────────────────────────────────────────
    segment_starts = [0] + breakpoints[:-1]   # start index of each segment
    segment_ends = breakpoints                # end index (exclusive)
    segment_months = [full_range[s] for s in segment_starts]

    segment_means = [
        signal[s:e].mean()
        for s, e in zip(segment_starts, segment_ends)
    ]
    segment_lengths = [e - s for s, e in zip(segment_starts, segment_ends)]

    baseline_rate = segment_means[0]
    # Avoid division-by-zero for restaurants with 0 reviews in early months
    safe_baseline = max(baseline_rate, 0.5)

    peak_idx  = int(np.argmax(segment_means))
    peak_rate = segment_means[peak_idx]
    magnitude = peak_rate / safe_baseline

    base["baseline_rate"] = round(baseline_rate, 3)
    base["peak_rate"] = round(peak_rate, 3)
    base["blowup_magnitude"] = round(magnitude, 2)
    base["blowup_month"] = str(segment_months[peak_idx])

    # ── Blowup criteria ───────────────────────────────────────────────────────
    # 1. Peak segment rate must be ≥ BLOWUP_RATIO × baseline
    if magnitude < BLOWUP_RATIO:
        base["blowup_reason"] = (
            f"peak_ratio_{magnitude:.1f}x_below_threshold_{BLOWUP_RATIO}x"
        )
        return base

    # 2. Peak segment must have lasted ≥ SUSTAINED_MONTHS
    sustained = segment_lengths[peak_idx] >= SUSTAINED_MONTHS
    base["post_peak_sustained"] = sustained
    if not sustained:
        base["blowup_reason"] = (
            f"spike_not_sustained_only_{segment_lengths[peak_idx]}_months"
        )
        return base

    # All criteria met
    base["blown_up"] = True
    base["blowup_reason"] = (
        f"peak_{peak_rate:.1f}_vs_baseline_{baseline_rate:.1f}_"
        f"ratio_{magnitude:.1f}x_sustained_{segment_lengths[peak_idx]}mo"
    )
    return base


# ── Main loop ─────────────────────────────────────────────────────────────────

results = []
businesses = list(df.groupby("business_id"))
n = len(businesses)

for i, (biz_id, grp) in enumerate(businesses):
    results.append(analyze_business(biz_id, grp))
    if (i + 1) % 50 == 0 or (i + 1) == n:
        blown = sum(r["blown_up"] for r in results)
        print(f"  {i+1:>4}/{n}  blown_up so far: {blown}")

# ── Output ────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
results_df["blown_up"] = results_df["blown_up"].fillna(False)
results_df["blowup_reason"] = results_df["blowup_reason"].fillna("none")
results_df["baseline_rate"] = results_df["baseline_rate"].fillna(0.0)
results_df["peak_rate"] = results_df["peak_rate"].fillna(0.0)
results_df["blowup_magnitude"] = results_df["blowup_magnitude"].fillna(1.0)
results_df["blowup_month"] = results_df["blowup_month"].fillna("")
results_df["post_peak_sustained"] = results_df["post_peak_sustained"].fillna(False)

results_df.to_csv(OUTPUT_PATH, index=False)

print(f"Full per-business blowup results written to {OUTPUT_PATH}")


# ── Summary ───────────────────────────────────────────────────────────────────

results_summary = pd.DataFrame(results)
eligible = results_summary[results_summary["eligible"]]
blown = results_summary[results_summary["blown_up"]]
not_blown = eligible[~eligible["blown_up"]]

print(f"""
────────────────────────────────
Results written to {OUTPUT_PATH}
────────────────────────────────
Total businesses : {len(results_summary)}
Eligible for PELT : {len(eligible)}
  Blown up : {len(blown)}  ({100*len(blown)/max(len(eligible),1):.2f}% of eligible)
  Not blown up : {len(not_blown)}
Ineligible (insufficient data): {len(results_summary)-len(eligible)}
""")