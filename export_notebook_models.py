"""
NOTEBOOK EXPORT CELL — paste this at the END of your Jupyter notebook
(after all cells have run, i.e. after Cell 10 / Scenario Engine).

It exports every artifact the eToro EOD bot needs:

  etoro_model_export/
    orig_model_h3.cbm          CatBoost model  (long leg, original features)
    af_model_h3.cbm            CatBoost model  (short leg, alpha-factory features)
    orig_pipeline_h3.pkl       {imp, sel, all_feature_cols, selected_feature_cols}
    af_pipeline_h3.pkl         same for alpha-factory
    regime_history.parquet     mkt_full DataFrame (all signal history)
    meta.json                  score rescale params, TOP_N, BOTTOM_N, split_date
"""

import json
import pickle
from pathlib import Path

EXPORT_DIR = Path("etoro_model_export")
EXPORT_DIR.mkdir(exist_ok=True)

H = 3  # horizon exported for trading

print("=" * 60)
print("EXPORTING MODELS FOR ETORO EOD BOT")
print("=" * 60)

# ── 1. CatBoost models ────────────────────────────────────────────────────────
for label, model in [("orig", orig_models[H]), ("af", af_models[H])]:
    path = str(EXPORT_DIR / f"{label}_model_h{H}.cbm")
    model.save_model(path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"  [{label}] model saved → {path}  ({size_kb:.0f} KB)")

# ── 2. Feature pipeline (imputer + selector + feature name lists) ─────────────
# We need to store:
#   imp            fitted SimpleImputer  (transform-only at inference)
#   sel            fitted SelectKBest    (transform-only at inference)
#   all_feat_cols  ordered list of column names fed INTO the imputer
#   sel_feat_cols  ordered list of column names AFTER selection (feat_names)

for label, data, feat_list in [
    ("orig", orig_data[H], [c for c in ORIG_FEATURES if c in df_scaled.columns]),
    ("af",   af_data[H],   [c for c in ALPHA_FEATURES if c in df_scaled.columns]),
]:
    # all_feat_cols must match exactly what was passed to imp.fit_transform()
    # In build_matrices: valid = [c for c in feature_cols if c in tr.columns]
    # feat_list above reconstructs that (df_scaled has same columns as tr after dropna)
    valid_in_scaled = [c for c in feat_list if c in df_scaled.columns]

    pipe = {
        "imp":           data["imp"],
        "sel":           data["sel"],
        "all_feat_cols": valid_in_scaled,        # ordered input to imputer
        "sel_feat_cols": data["feat_names"],     # selected feature names (post selector)
    }
    path = EXPORT_DIR / f"{label}_pipeline_h{H}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    n_all = len(valid_in_scaled)
    n_sel = len(data["feat_names"])
    print(f"  [{label}] pipeline saved → {path}  ({n_all} → {n_sel} features)")

# ── 3. Regime history ─────────────────────────────────────────────────────────
reg_path = EXPORT_DIR / "regime_history.parquet"
mkt_full.to_parquet(reg_path)
print(f"  [regime] history saved → {reg_path}  ({len(mkt_full)} dates)")

# ── 4. Meta / config ──────────────────────────────────────────────────────────
meta = {
    # Score rescaling (computed in scenario engine cell)
    "score_min": float(_score_min),
    "score_max": float(_score_max),
    # Strategy parameters
    "top_n":       int(TOP_N),
    "bottom_n":    int(BOTTOM_N),
    "horizon":     int(H),
    # Date reference
    "split_date":  str(split_date.date()),
    # MaxLev_Bear25 scenario parameters (from Cell 11 SCENARIOS dict)
    "scenario": {
        "name":          "MaxLev_Bear25",
        "long_base":     0.85,
        "bull_long":     1.00,
        "bear_long":     0.25,
        "lev_bull_max":  10.0,
        "lev_neutral":   5.0,
        "lev_bear_min":  0.5,
        "vol_target":    0.20,
        "dd_delever":    -0.15,
        "max_daily_loss": -0.05,
        "top_n":         int(TOP_N),
        "bottom_n":      int(BOTTOM_N),
    },
    # Regime detection parameters (match notebook Cell 9)
    "regime": {
        "lookback":     252,
        "bull_pct":     67,
        "bear_pct":     33,
        "min_thresh_hist": 60,
    },
    # VOL_LOOKBACK used in risk controls
    "vol_lookback": int(VOL_LOOKBACK),
}

meta_path = EXPORT_DIR / "meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  [meta] saved → {meta_path}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Export complete  →  {EXPORT_DIR}/")
print(f"{'='*60}")
for p in sorted(EXPORT_DIR.iterdir()):
    print(f"  {p.name:<45s}  {p.stat().st_size / 1024:>7.1f} KB")

print(f"""
Next steps:
  1. Copy the entire '{EXPORT_DIR}/' directory to your server.
  2. Set environment variables:
       FMP_API_KEY    = <your FMP key>
       ETORO_API_KEY  = <your eToro API key>
       ETORO_ACCOUNT  = <your eToro account ID>
  3. Run the EOD bot 10 min before market close:
       python etoro_eod_trader.py --export-dir etoro_model_export/
  4. Or schedule it:
       50 15 * * 1-5  cd /path/to/repo && python etoro_eod_trader.py
""")
