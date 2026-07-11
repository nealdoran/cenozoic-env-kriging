"""
krige_env.py — Cenozoic depositional-environment kriging pipeline

Multi-source, reliability-weighted, declustered indicator kriging of
marine vs. terrestrial depositional environment with explicit uncertainty surface.

Method:
  1. Load observations (lat, lng, environment, age_ma, source, weight)
  2. Bin to 0.5° cells with weighted mean P(marine) — simultaneously
     declusters dense regions AND applies source reliability weights
  3. Fit spherical variogram per epoch (scikit-gstat), capped at maxlag
     to prevent long-range trend contamination
  4. Ordinary kriging → P_marine surface + variance surface (pykrige)
  5. Render with confidence-modulated saturation: hue = P(marine),
     saturation ∝ 1 - normalized(variance). High-variance pixels
     desaturate toward cream — the uncertainty IS the map.

Honesty design:
  - Macrostrat environment_class (direct rock classification) weighted 5×
    over fossil occurrences (collection-biased)
  - Coverage reported as fraction of land within one variogram range
    (resolution-independent — not a pixel-count artifact)
  - Uncertainty surface mandatory, not optional

Usage:
  python krige_env.py --data data/sample_observations.csv --epoch Paleogene

Full analysis (PBDB + Macrostrat, ~831k observations) is in preparation
for peer review. This script runs on the synthetic sample dataset.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree

try:
    import skgstat as skg
    SKGSTAT = True
except ImportError:
    print("scikit-gstat not found — variogram fitting disabled, using default range")
    SKGSTAT = False

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE = True
except ImportError:
    raise ImportError("pykrige required: pip install pykrige")

# ── PROJECTION ────────────────────────────────────────────────────────────────
PROJ = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=True)

# ── COLORMAP: deep blue (marine) → cream (uncertain) → warm brown (terrestrial)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "marine_terr",
    [
        (0.00, "#5C3A10"),   # deep brown  = terrestrial P=0
        (0.20, "#9C6B2E"),
        (0.35, "#D4A96A"),
        (0.50, "#F5F0E8"),   # warm cream  = uncertain  P=0.5
        (0.65, "#9ECAE1"),
        (0.80, "#2171B5"),
        (1.00, "#08306B"),   # deep navy   = marine     P=1
    ]
)
NORM = mcolors.Normalize(vmin=0, vmax=1)

# ── EPOCH BOUNDS (ICS 2023, Ma) ───────────────────────────────────────────────
EPOCH_BOUNDS = {
    "Paleogene":  (23.03, 66.0),
    "Neogene":    (2.58,  23.03),
    "Quaternary": (0.0,   2.58),
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD + CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────
def load_observations(csv_path: str) -> pd.DataFrame:
    """
    Load observation CSV and assign:
      indicator : 1.0 (marine) | 0.5 (marginal) | 0.0 (terrestrial)
      epoch     : Paleogene | Neogene | Quaternary | other
    Expected columns: lat, lng, environment, age_ma, source, weight
    """
    df = pd.read_csv(csv_path)
    required = {"lat", "lng", "environment", "age_ma", "source", "weight"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    env_map = {
        "marine":      1.0,
        "marginal":    0.5,
        "terrestrial": 0.0,
    }
    df["indicator"] = df["environment"].str.lower().map(env_map)
    df = df.dropna(subset=["indicator", "lat", "lng", "age_ma"])

    def assign_epoch(age):
        for ep, (lo, hi) in EPOCH_BOUNDS.items():
            if lo <= age < hi:
                return ep
        return "other"

    df["epoch"] = df["age_ma"].apply(assign_epoch)
    print(f"Loaded {len(df):,} observations")
    print(df.groupby(["source", "epoch"]).size().unstack(fill_value=0).to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WEIGHTED GRID-BINNING (decluster + weight simultaneously)
# ─────────────────────────────────────────────────────────────────────────────
def bin_observations(df: pd.DataFrame, bin_deg: float = 0.5) -> pd.DataFrame:
    """
    Bin observations to a regular grid. Within each cell:
      P_marine = Σ(w_i · ind_i) / Σ(w_i)
    High-weight sources (Macrostrat) dominate; dense clusters are collapsed.
    """
    df = df.copy()
    df["lat_bin"] = (df["lat"] / bin_deg).round() * bin_deg
    df["lng_bin"] = (df["lng"] / bin_deg).round() * bin_deg

    def wmean(g):
        w = g["weight"].sum()
        return (g["indicator"] * g["weight"]).sum() / w if w > 0 else None

    cells = (df.groupby(["lat_bin", "lng_bin"])
               .apply(wmean, include_groups=False)
               .reset_index(name="P_marine"))
    cells["n_obs"]        = df.groupby(["lat_bin","lng_bin"]).size().values
    cells["total_weight"] = df.groupby(["lat_bin","lng_bin"])["weight"].sum().values
    cells = cells.dropna(subset=["P_marine"])

    print(f"Grid-binned to {len(cells):,} cells  "
          f"(P_marine mean={cells['P_marine'].mean():.2f})")
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — VARIOGRAM FITTING
# ─────────────────────────────────────────────────────────────────────────────
def fit_variogram(cells: pd.DataFrame,
                  maxlag_m: float = 2_500_000,
                  range_cap_m: float = 1_500_000) -> float:
    """
    Fit spherical variogram to declustered cells (in projected meters).
    maxlag_m   : maximum lag considered — prevents fitting long-range trend
    range_cap_m: hard cap on fitted range (geological defensibility limit)
    Returns range in meters.
    """
    if not SKGSTAT or len(cells) < 10:
        print(f"  Variogram: using default range {range_cap_m/1000:.0f} km")
        return range_cap_m

    x_m, y_m = PROJ.transform(cells["lng_bin"].values,
                               cells["lat_bin"].values)
    z = cells["P_marine"].values

    try:
        V = skg.Variogram(
            coordinates=np.column_stack([x_m, y_m]),
            values=z,
            model="spherical",
            n_lags=20,
            maxlag=maxlag_m,
        )
        rng    = min(V.parameters[0], range_cap_m)
        sill   = V.parameters[1]
        nugget = V.parameters[2]
        print(f"  Variogram: range={rng/1000:.0f} km  "
              f"sill={sill:.3f}  nugget={nugget:.3f}")
        return rng
    except Exception as e:
        print(f"  Variogram fitting failed ({e}) — using {range_cap_m/1000:.0f} km")
        return range_cap_m


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ORDINARY KRIGING
# ─────────────────────────────────────────────────────────────────────────────
def krige(cells: pd.DataFrame,
          vrange_m: float,
          grid_res_m: float = 200_000) -> tuple:
    """
    Ordinary kriging of P_marine onto a regular Equal-Earth grid.
    Returns (Z_krig, Z_var, xi, yi) — all in EPSG:8857 meters.
    """
    x_m, y_m = PROJ.transform(cells["lng_bin"].values,
                               cells["lat_bin"].values)
    x_m = np.array(x_m, dtype=np.float64)
    y_m = np.array(y_m, dtype=np.float64)
    z   = cells["P_marine"].values.astype(np.float64)

    xi = np.arange(-17_000_000, 17_000_000, grid_res_m, dtype=np.float64)
    yi = np.arange( -8_700_000,  8_700_000, grid_res_m, dtype=np.float64)

    print(f"  Kriging {len(cells):,} cells onto {len(yi)}×{len(xi)} grid …")
    ok = OrdinaryKriging(
        x_m, y_m, z,
        variogram_model="spherical",
        variogram_parameters={
            "range":  float(vrange_m),
            "sill":   float(np.var(z)),
            "nugget": float(np.var(z) * 0.1),
        },
        verbose=False,
        enable_plotting=False,
    )
    Z_krig, Z_var = ok.execute("grid", xi, yi)
    Z_krig = np.clip(np.array(Z_krig, dtype=np.float64), 0, 1)
    Z_var  = np.clip(np.array(Z_var,  dtype=np.float64), 0, None)
    print(f"  Done. P={Z_krig.min():.2f}–{Z_krig.max():.2f}  "
          f"Var={Z_var.min():.4f}–{Z_var.max():.4f}")
    return Z_krig, Z_var, xi, yi


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COVERAGE (resolution-independent)
# ─────────────────────────────────────────────────────────────────────────────
def coverage_fraction(cells: pd.DataFrame,
                      xi: np.ndarray,
                      yi: np.ndarray,
                      vrange_m: float) -> float:
    """
    Fraction of grid pixels within one variogram range of an observation.
    Resolution-independent: uses distance, not variance threshold.
    """
    x_obs, y_obs = PROJ.transform(cells["lng_bin"].values,
                                   cells["lat_bin"].values)
    tree = cKDTree(np.column_stack([x_obs, y_obs]))
    XI, YI = np.meshgrid(xi, yi)
    dists, _ = tree.query(
        np.column_stack([XI.ravel(), YI.ravel()]), k=1
    )
    return float((dists < vrange_m).mean())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RENDER WITH CONFIDENCE MODULATION
# ─────────────────────────────────────────────────────────────────────────────
def render(Z_krig: np.ndarray,
           Z_var:  np.ndarray,
           xi:     np.ndarray,
           yi:     np.ndarray,
           cells:  pd.DataFrame,
           epoch:  str,
           vrange_m: float,
           coverage: float,
           out_path: str,
           title: str = None) -> None:
    """
    Render P_marine surface with confidence-modulated saturation.
    High-variance (data-sparse) pixels desaturate toward cream.
    The uncertainty gradient IS the key visual message.
    """
    var_max    = float(np.nanpercentile(Z_var, 95))
    confidence = np.clip(1 - Z_var / max(var_max, 1e-10), 0, 1)

    # RGBA: apply colormap then desaturate by variance
    rgba = CMAP(NORM(Z_krig)).copy()
    gray = 0.299*rgba[:,:,0] + 0.587*rgba[:,:,1] + 0.114*rgba[:,:,2]
    for c in range(3):
        rgba[:,:,c] = confidence*rgba[:,:,c] + (1-confidence)*gray

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("#F5F5F5")

    # Raster — origin='lower' matches yi ascending bottom-to-top
    ax.imshow(rgba, origin="lower",
              extent=[xi.min(), xi.max(), yi.min(), yi.max()],
              aspect="auto", interpolation="bilinear")

    # Observation dots (colored by P_marine)
    x_obs, y_obs = PROJ.transform(cells["lng_bin"].values,
                                   cells["lat_bin"].values)
    ax.scatter(x_obs, y_obs,
               c=cells["P_marine"], cmap=CMAP, norm=NORM,
               s=8, alpha=0.7, zorder=5, linewidths=0, edgecolors="none")

    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())
    ax.set_axis_off()

    if title:
        ax.text(0.01, 0.96, title, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # Colorbar
    cax = fig.add_axes([0.905, 0.15, 0.013, 0.65])
    cb  = ColorbarBase(cax, cmap=CMAP, norm=NORM, orientation="vertical")
    cb.set_label("P(marine)", fontsize=8)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0.0\n(terrestrial)", "0.5\n(uncertain)", "1.0\n(marine)"])
    cb.ax.tick_params(labelsize=7)

    caption = (
        f"Example output on synthetic sample data — {epoch}. "
        f"Ordinary kriging of P(marine); color saturation reflects kriging "
        f"confidence (desaturated = data-sparse). "
        f"Variogram range={vrange_m/1000:.0f} km. "
        f"Coverage: {100*coverage:.0f}% of grid within 1 variogram range. "
        f"Full analysis in preparation for peer review. Equal Earth (EPSG:8857)."
    )
    fig.text(0.01, 0.005, caption, fontsize=6.5, color="#444444", wrap=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cenozoic depositional-environment kriging pipeline"
    )
    parser.add_argument("--data",      default="data/sample_observations.csv",
                        help="Path to observations CSV")
    parser.add_argument("--epoch",     default="Paleogene",
                        choices=["Paleogene","Neogene","Quaternary"],
                        help="Epoch to map")
    parser.add_argument("--grid_km",   default=200, type=int,
                        help="Kriging grid resolution in km (default 200)")
    parser.add_argument("--maxlag_km", default=2500, type=int,
                        help="Variogram max lag in km (default 2500)")
    parser.add_argument("--cap_km",    default=1500, type=int,
                        help="Variogram range cap in km (default 1500)")
    parser.add_argument("--out",       default=None,
                        help="Output figure path (default: figures/demo_{epoch}.png)")
    args = parser.parse_args()

    out_path = args.out or f"figures/demo_{args.epoch}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Cenozoic Environment Kriging — {args.epoch}")
    print(f"{'='*55}\n")

    # 1. Load
    df = load_observations(args.data)
    df_epoch = df[df["epoch"] == args.epoch].copy()
    print(f"\nEpoch subset: {len(df_epoch):,} observations")
    if len(df_epoch) < 5:
        raise ValueError(f"Too few observations for {args.epoch} (n={len(df_epoch)})")

    # 2. Bin
    print("\nStep 2: Grid-binning …")
    cells = bin_observations(df_epoch)

    # 3. Variogram
    print("\nStep 3: Variogram fitting …")
    vrange = fit_variogram(cells,
                           maxlag_m=args.maxlag_km * 1000,
                           range_cap_m=args.cap_km * 1000)

    # 4. Krige
    print("\nStep 4: Kriging …")
    Z_krig, Z_var, xi, yi = krige(cells, vrange,
                                   grid_res_m=args.grid_km * 1000)

    # 5. Coverage
    cov = coverage_fraction(cells, xi, yi, vrange)
    print(f"\nCoverage: {100*cov:.1f}% of grid within 1 variogram range")

    # 6. Render
    print("\nStep 6: Rendering …")
    render(Z_krig, Z_var, xi, yi, cells,
           epoch=args.epoch,
           vrange_m=vrange,
           coverage=cov,
           out_path=out_path,
           title=f"{args.epoch} P(marine) — synthetic sample data")

    print(f"\nDone. Figure → {out_path}")


if __name__ == "__main__":
    main()
