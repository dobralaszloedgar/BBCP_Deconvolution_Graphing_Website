import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import pandas as pd
import streamlit as st


def run_deconvolution(
        data_array,
        calib_array,
        mw_lim,
        y_lim,
        n_peaks,
        plot_sum,
        manual_peaks,
        peaks_are_mw,
        peak_names,
        peak_width_range,
        baseline_method,
        baseline_ranges
):
    """
    Perform GPC chromatogram deconvolution with Gaussian peak fitting.

    Parameters:
    data_array: np.array - Chromatogram data [retention_time, response]
    calib_array: np.array - Calibration data [retention_time, log10(MW)]
    mw_lim: list - Molecular weight limits [min, max]
    y_lim: list - Y-axis limits [min, max]
    n_peaks: int - Number of peaks to fit
    plot_sum: bool - Whether to plot sum of fitted peaks
    manual_peaks: list - Manual peak positions (MW or RT)
    peaks_are_mw: bool - True if manual peaks are MW values
    peak_names: list - Names for each peak
    peak_width_range: list - [min, max] width range for peak fitting
    baseline_method: str - 'flat', 'linear', or 'quadratic'
    baseline_ranges: list - MW ranges for baseline correction

    Returns:
    fig: matplotlib.figure.Figure - Deconvolution plot
    results_df: pd.DataFrame - Peak analysis results
    """

    # 1. Build calibration interpolators
    rt_cal, logmw_cal = calib_array[:, 0], calib_array[:, 1]
    f_logmw = interp1d(rt_cal, logmw_cal, kind='linear', fill_value='extrapolate')
    f_rt = interp1d(logmw_cal, rt_cal, kind='linear', fill_value='extrapolate')

    def mw_to_rt(mw):
        return f_rt(np.log10(mw))

    # 2. Convert MW limits to RT limits
    rt_min = mw_to_rt(mw_lim[1])
    rt_max = mw_to_rt(mw_lim[0])
    rt_lim = [rt_min, rt_max]

    # 3. Extract and normalize chromatogram data
    x_rt = data_array[:, 0].astype(float)
    y_raw = data_array[:, 1].astype(float)

    # Normalize within RT window
    mask = (x_rt >= rt_lim[0]) & (x_rt <= rt_lim[1])
    y_norm = y_raw / np.max(y_raw[mask]) if np.any(mask) else y_raw

    # 4. Baseline correction
    x_mw = 10 ** f_logmw(x_rt)  # Convert RT to MW for baseline ranges

    def correct_baseline(x_rt, y, x_mw):
        ref_points = []
        required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}[baseline_method]

        if len(baseline_ranges) != required_ranges:
            raise ValueError(f"{baseline_method} baseline requires {required_ranges} ranges")

        for bl_range in baseline_ranges:
            mask = (x_mw >= bl_range[0]) & (x_mw <= bl_range[1])
            if np.any(mask):
                ref_points.append((np.mean(x_rt[mask]), np.mean(y[mask])))

        x_ref, y_ref = zip(*ref_points)
        degree = {'flat': 0, 'linear': 1, 'quadratic': 2}[baseline_method]
        coeffs = np.polyfit(x_ref, y_ref, degree)
        baseline = np.polyval(coeffs, x_rt)
        return y - baseline, baseline

    y_corrected, baseline = correct_baseline(x_rt, y_norm, x_mw)

    # 5. Peak detection
    if manual_peaks:
        x_peaks_rt = [mw_to_rt(p) if peaks_are_mw else p for p in manual_peaks]
        n_peaks = len(x_peaks_rt)
    else:
        indices, _ = find_peaks(y_corrected, distance=200, width=50)
        peak_heights = y_corrected[indices]
        if len(peak_heights) >= n_peaks:
            top_indices = np.argsort(peak_heights)[-n_peaks:][::-1]
            x_peaks_rt = x_rt[indices[top_indices]]
        else:
            x_peaks_rt = x_rt[indices]
            n_peaks = len(x_peaks_rt)

    # 6. Gaussian fitting
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    best_fit = None
    best_params = None
    best_residual = np.inf

    for width in range(peak_width_range[0], peak_width_range[1]):
        y_temp = y_corrected.copy()
        gaussians = []
        params_list = []

        try:
            for peak_rt in x_peaks_rt:
                idx = np.argmin(np.abs(x_rt - peak_rt))
                start = max(0, idx - width)
                end = min(len(x_rt), idx + width)

                p0 = [y_temp[idx], peak_rt, 0.1]
                params, _ = curve_fit(gaussian, x_rt[start:end], y_temp[start:end], p0=p0)

                fit_curve = gaussian(x_rt, *params)
                gaussians.append(fit_curve)
                params_list.append(params)

                y_temp -= fit_curve

            residual = np.sum(np.abs(y_temp))
            if residual < best_residual:
                best_residual = residual
                best_fit = gaussians
                best_params = params_list
        except RuntimeError:
            continue

    if best_params is None:
        raise RuntimeError("Peak fitting failed for all width attempts")

    # 7. Calculate results
    mus = [p[1] for p in best_params]
    mw_values = 10 ** f_logmw(np.array(mus))

    areas = [trapezoid(g, x_rt) for g in best_fit]
    total_area = sum(areas)
    area_percentages = [(area / total_area) * 100 for area in areas]

    # Sort by molecular weight (descending)
    sort_indices = np.argsort(mw_values)[::-1]
    mw_values = mw_values[sort_indices]
    area_percentages = np.array(area_percentages)[sort_indices]
    best_fit = np.array(best_fit)[sort_indices]

    # Prepare peak names
    peak_names = (peak_names + [f"Peak {i + 1}" for i in range(len(mw_values))])[:len(mw_values)]

    # 8. Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_mw, y_corrected, color='#ef476f', linewidth=2, label='Baseline-corrected')

    colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c']
    for i, (fit, pct) in enumerate(zip(best_fit, area_percentages)):
        ax.plot(x_mw, fit, color=colors[i % len(colors)],
                label=f'{peak_names[i]}: {pct:.1f}%')

    if plot_sum:
        sum_fit = np.sum(best_fit, axis=0)
        ax.plot(x_mw, sum_fit, '--', color='black', linewidth=1.5, label='Sum of Gaussians')

    ax.set_xscale('log')
    ax.set_xlim(mw_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Molecular weight (g/mol)")
    ax.set_ylabel("Normalized Response")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()

    # 9. Create results table
    results_df = pd.DataFrame({
        'Peak': peak_names,
        'Mn (g/mol)': mw_values.astype(int),
        'Area %': np.round(area_percentages, 1)
    })

    return fig, results_df