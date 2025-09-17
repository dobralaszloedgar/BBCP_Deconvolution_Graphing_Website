# gaussian_deconvolution.py
# Streamlit app with centered layout using set_page_config and a column wrapper (no CSS).

from Deconvolution import *  # keep existing project import as-is

import streamlit as st
import numpy as np
import io
import os
import sys
import json
import base64
import tempfile
import requests

# Optional scientific stack for fitting/smoothing
try:
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Matplotlib for plotting
import matplotlib.pyplot as plt


# -------------------------------------------------
# Page configuration: request centered layout first
# -------------------------------------------------
try:
    st.set_page_config(
        page_title="Gaussian Deconvolution",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
except Exception:
    # Streamlit only allows set_page_config once; ignore if a launcher already set it.
    pass


# -------------------------------------------------
# Lightweight centering helper (no CSS)
# -------------------------------------------------
from contextlib import contextmanager

@contextmanager
def centered():
    """
    Render all enclosed elements in the middle column so the app looks centered
    even if the parent app forced a wide layout.
    """
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        yield


# -------------------------------------------------
# Navigation helpers (kept from original)
# -------------------------------------------------
def _clear_query_params_and_rerun():
    """
    Clear URL query params and rerun the app to reset state across navigation hops.
    Works across Streamlit versions by trying the new API first and then the legacy API.
    """
    try:
        # New API (Streamlit >= 1.30)
        st.query_params.clear()
    except Exception:
        # Old API
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
    st.rerun()


# -------------------------------------------------
# Math helpers for fitting
# -------------------------------------------------
def _gaussian(x, amp, mu, sigma):
    sigma = np.maximum(np.asarray(sigma), 1e-12)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _multi_gaussian(x, *params):
    # params = [A1, mu1, s1, A2, mu2, s2, ...]
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        A, mu, s = params[i : i + 3]
        y = y + _gaussian(x, A, mu, s)
    return y


def _poly_baseline(x, y, deg=1):
    deg = int(max(0, deg))
    if deg == 0:
        return np.full_like(y, np.median(y))
    coefs = np.polyfit(x, y, deg)
    return np.polyval(coefs, x)


def _initial_guesses_from_peaks(x, y, n_peaks: int):
    # Heuristics: use find_peaks to locate candidates; if not enough, pad uniformly.
    if not _HAS_SCIPY:
        # Fallback: uniform guesses across domain
        amps = np.full(n_peaks, max(1e-3, (np.max(y) - np.min(y)) / max(1, n_peaks)))
        mus = np.linspace(np.min(x), np.max(x), n_peaks + 2)[1:-1]
        sigmas = np.full(n_peaks, (np.max(x) - np.min(x)) / (8.0 * n_peaks))
        return np.ravel(np.column_stack([amps, mus, sigmas]))

    # With SciPy: detect peaks
    safe_y = np.asarray(y, dtype=float)
    safe_y = safe_y - np.nanmin(safe_y)
    peak_idx, _ = find_peaks(safe_y, distance=max(1, len(y) // (n_peaks * 3)))
    if len(peak_idx) == 0:
        # No detected peaks; fall back to uniform guesses
        amps = np.full(n_peaks, max(1e-3, (np.max(y) - np.min(y)) / max(1, n_peaks)))
        mus = np.linspace(np.min(x), np.max(x), n_peaks + 2)[1:-1]
        sigmas = np.full(n_peaks, (np.max(x) - np.min(x)) / (8.0 * n_peaks))
        return np.ravel(np.column_stack([amps, mus, sigmas]))

    # Sort peaks by amplitude and take top n
    top = np.argsort(safe_y[peak_idx])[::-1][:n_peaks]
    sel = np.sort(peak_idx[top])
    # Estimate widths as sigma via peak widths at half prominence
    try:
        pw, _, _, _ = peak_widths(safe_y, sel, rel_height=0.5)
        # Convert width (in sample index) to sigma roughly: FWHM ≈ 2.355*sigma => sigma ≈ width/2.355
        est_sigma = np.clip(pw / 2.355, 1.0, max(2.0, len(x) / (10 * n_peaks)))
        # Map index-scale widths into x-scale based on local dx
        dx = np.mean(np.diff(np.asarray(x, dtype=float)))
        sigmas = est_sigma * max(dx, 1e-12)
    except Exception:
        sigmas = np.full(len(sel), (np.max(x) - np.min(x)) / (8.0 * max(1, n_peaks)))

    amps = np.maximum(safe_y[sel], 1e-6)
    mus = np.asarray(x)[sel]
    guesses = np.ravel(np.column_stack([amps, mus, sigmas]))
    # Pad if fewer detected than requested
    if len(sel) < n_peaks:
        pad = n_peaks - len(sel)
        amps_pad = np.full(pad, np.median(amps) if amps.size else 1.0)
        mus_pad = np.linspace(np.min(x), np.max(x), pad + 2)[1:-1]
        sigmas_pad = np.full(pad, (np.max(x) - np.min(x)) / (8.0 * n_peaks))
        guesses = np.concatenate([guesses, np.ravel(np.column_stack([amps_pad, mus_pad, sigmas_pad]))])
    return guesses


def _fit_multi_gaussian(x, y, n_peaks: int, bounds_mode: str = "loose"):
    """
    Fit a sum of Gaussians to y(x).
    bounds_mode:
      - 'none': unbounded
      - 'loose': amp >= 0, sigma in (1e-6, span), mu within [xmin-10%, xmax+10%]
      - 'tight': amp >= 0, sigma in (1e-6, span/4), mu within [xmin, xmax]
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    assert len(x) == len(y), "x and y must have same length"
    p0 = _initial_guesses_from_peaks(x, y, n_peaks)

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy is required for non-linear Gaussian fitting but is not available.")

    xmin, xmax = np.min(x), np.max(x)
    span = max(1e-12, xmax - xmin)

    if bounds_mode == "none":
        bounds = (-np.inf, np.inf)
    else:
        if bounds_mode == "tight":
            mu_lo, mu_hi = xmin, xmax
            sigma_hi = max(1e-4, span / 4.0)
        else:  # loose
            mu_lo, mu_hi = xmin - 0.1 * span, xmax + 0.1 * span
            sigma_hi = max(1e-4, span)
        lo = []
        hi = []
        for i in range(n_peaks):
            lo.extend([0.0, mu_lo, 1e-6])
            hi.extend([np.inf, mu_hi, sigma_hi])
        bounds = (np.array(lo, dtype=float), np.array(hi, dtype=float))

    popt, pcov = curve_fit(_multi_gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)
    y_fit = _multi_gaussian(x, *popt)
    return popt, pcov, y_fit


# -------------------------------------------------
# App
# -------------------------------------------------
def main():
    # Sidebar controls remain in the sidebar
    with st.sidebar:
        st.header("Controls")
        data_mode = st.radio("Data source", ["Upload CSV", "Synthetic"], index=0, horizontal=True)
        smoothing = st.checkbox("Apply smoothing (Savitzky-Golay)", value=False)
        smooth_win = st.number_input("S-G window length (odd)", min_value=3, value=15, step=2, help="Must be odd.")
        smooth_poly = st.number_input("S-G polyorder", min_value=1, value=3, step=1)
        baseline_deg = st.number_input("Baseline degree", min_value=0, value=0, step=1)
        n_peaks = st.number_input("Number of peaks", min_value=1, value=2, step=1)
        bounds_mode = st.selectbox("Bounds", ["loose", "tight", "none"], index=0)
        do_fit = st.button("Run deconvolution")
        st.markdown("---")
        if st.button("Clear URL params"):
            _clear_query_params_and_rerun()

    # All main-body content is wrapped in the centered column
    with centered():
        st.title("Gaussian Deconvolution 🔬")
        st.caption("Fit sums of Gaussian peaks with optional smoothing and baseline correction.")

        # Data acquisition
        x = None
        y = None
        dataset_name = None

        if data_mode == "Upload CSV":
            up = st.file_uploader("Upload CSV with two numeric columns (x,y)", type=["csv"])
            if up is not None:
                import pandas as pd
                try:
                    df = pd.read_csv(up)
                    # Heuristic: first two numeric columns
                    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                    if len(num_cols) >= 2:
                        x = df[num_cols[0]].to_numpy()
                        y = df[num_cols[1]].to_numpy()
                        dataset_name = up.name
                    else:
                        st.error("CSV must contain at least two numeric columns.")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")
        else:
            # Synthetic data generator
            st.subheader("Synthetic data")
            rng_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
            x_min, x_max, n_pts = 0.0, 100.0, 800
            x = np.linspace(x_min, x_max, n_pts)
            with np.random.default_rng(int(rng_seed)):
                # Default two peaks
                means = st.text_input("Means (comma)", "30, 70")
                sigmas = st.text_input("Sigmas (comma)", "5, 7")
                amps = st.text_input("Amplitudes (comma)", "1.0, 0.8")
                noise = st.number_input("Noise std", min_value=0.0, value=0.03, step=0.01)

                try:
                    mus = np.array([float(s.strip()) for s in means.split(",") if s.strip()])
                    sgs = np.array([float(s.strip()) for s in sigmas.split(",") if s.strip()])
                    aps = np.array([float(s.strip()) for s in amps.split(",") if s.strip()])
                    k = min(len(mus), len(sgs), len(aps))
                    mus, sgs, aps = mus[:k], sgs[:k], aps[:k]
                    y = np.zeros_like(x, dtype=float)
                    for A, m, s in zip(aps, mus, sgs):
                        y += _gaussian(x, A, m, s)
                    y += np.random.normal(0.0, noise, size=x.shape)
                    dataset_name = "synthetic"
                    st.info(f"Synthetic with {k} peaks.")
                except Exception as e:
                    st.error(f"Failed to parse synthetic parameters: {e}")
                    x, y = None, None

        if x is None or y is None:
            st.stop()

        # Preprocessing: smoothing and baseline
        y_proc = y.copy()
        if smoothing and _HAS_SCIPY:
            try:
                win = int(max(3, smooth_win if smooth_win % 2 == 1 else smooth_win + 1))
                poly = int(max(1, min(smooth_poly, win - 1)))
                y_proc = savgol_filter(y_proc, window_length=win, polyorder=poly, mode="interp")
            except Exception as e:
                st.warning(f"Savitzky-Golay smoothing failed: {e}")
        elif smoothing and not _HAS_SCIPY:
            st.warning("SciPy not available; smoothing skipped.")

        if baseline_deg >= 0:
            try:
                base = _poly_baseline(x, y_proc, deg=int(baseline_deg))
                y_proc = y_proc - base
            except Exception as e:
                st.warning(f"Baseline correction failed: {e}")

        # Plot raw/preprocessed
        fig0, ax0 = plt.subplots(figsize=(7, 3))
        ax0.plot(x, y, color="#8da0cb", lw=1.5, label="Raw")
        ax0.plot(x, y_proc, color="#fc8d62", lw=1.2, label="Processed")
        ax0.set_xlabel("x")
        ax0.set_ylabel("intensity")
        ax0.legend(loc="best")
        ax0.grid(alpha=0.2)
        st.pyplot(fig0, clear_figure=True)

        # Fitting
        if do_fit:
            try:
                popt, pcov, y_fit = _fit_multi_gaussian(x, y_proc, int(n_peaks), bounds_mode=bounds_mode)
                resid = y_proc - y_fit

                # Unpack params
                comps = []
                for i in range(0, len(popt), 3):
                    A, mu, s = popt[i : i + 3]
                    comps.append({"peak": i // 3 + 1, "amplitude": float(A), "mu": float(mu), "sigma": float(s)})

                # Plot fit + components
                fig1, ax1 = plt.subplots(figsize=(7, 4))
                ax1.plot(x, y_proc, color="#555", lw=1.0, label="Processed")
                ax1.plot(x, y_fit, color="#1b9e77", lw=2.0, label="Fit")
                # Components
                for j, c in enumerate(comps):
                    ax1.plot(x, _gaussian(x, c["amplitude"], c["mu"], c["sigma"]), lw=1.2, alpha=0.8, label=f"Peak {j+1}")
                ax1.set_xlabel("x")
                ax1.set_ylabel("intensity")
                ax1.legend(ncol=2, fontsize=9)
                ax1.grid(alpha=0.2)
                st.pyplot(fig1, clear_figure=True)

                # Residuals
                fig2, ax2 = plt.subplots(figsize=(7, 2.6))
                ax2.plot(x, resid, color="#d95f02", lw=1.0)
                ax2.axhline(0.0, color="#999", lw=0.8, ls="--")
                ax2.set_xlabel("x")
                ax2.set_ylabel("residual")
                ax2.grid(alpha=0.2)
                st.pyplot(fig2, clear_figure=True)

                # Results table
                import pandas as pd
                df_params = pd.DataFrame(comps, columns=["peak", "amplitude", "mu", "sigma"])
                st.subheader("Fitted parameters")
                st.dataframe(df_params, use_container_width=True)

                # Downloads
                out = pd.DataFrame({"x": x, "y_processed": y_proc, "y_fit": y_fit, "residual": resid})
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    "Download fit CSV",
                    data=buf.getvalue().encode("utf-8"),
                    file_name=f"gaussian_fit_{dataset_name or 'data'}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Fitting failed: {e}")
                if not _HAS_SCIPY:
                    st.info("Tip: install SciPy to enable non-linear fitting (curve_fit).")


if __name__ == "__main__":
    main()
