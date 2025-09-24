import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import streamlit as st

# Optional baseline correction via pybaselines
try:
    from pybaselines import Baseline

    PYBASELINES_AVAILABLE = True
except ImportError:
    PYBASELINES_AVAILABLE = False
    st.warning("pybaselines not installed. ARPLS baseline correction will not be available.")


def setup_custom_fonts():
    """Add custom fonts from ./fonts if present."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(script_dir, 'fonts')
        if os.path.exists(fonts_dir):
            font_files = fm.findSystemFonts(fontpaths=[fonts_dir])
            for font_file in font_files:
                fm.fontManager.addfont(font_file)
            fm._load_fontmanager(try_read_cache=False)
            return True
        return False
    except Exception as e:
        st.warning(f"Could not set up custom fonts: {str(e)}")
        return False


setup_custom_fonts()


def baseline_correction(x_rt, y, x_plot, method='None', baseline_ranges=[]):
    """
    Return y_corrected and baseline.
    """
    if method == 'None':
        return y, np.zeros_like(y)

    if method == 'arpls':
        if not PYBASELINES_AVAILABLE:
            raise ImportError("pybaselines is required for arpls baseline correction")
        baseline_fitter = Baseline()
        baseline = baseline_fitter.arpls(y, lam=1e12, tol=1e-4, max_iter=100)[0]
        return y - baseline, baseline

    ref_points = []
    required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}.get(method, 0)
    if len(baseline_ranges) != required_ranges:
        raise ValueError(f"{method} method requires {required_ranges} baseline ranges")

    for bl_range in baseline_ranges:
        mask = (x_plot >= bl_range[0]) & (x_plot <= bl_range[1])
        if np.sum(mask) == 0:
            raise ValueError(f"No data points in baseline range {bl_range}")
        x_ref, y_ref = np.mean(x_rt[mask]), np.mean(y[mask])
        ref_points.append((x_ref, y_ref))

    x_vals = [p[0] for p in ref_points]
    y_vals = [p[1] for p in ref_points]

    if method == 'flat':
        baseline = np.full_like(y, np.mean(y_vals))
    elif method == 'linear':
        coeffs = np.polyfit(x_vals, y_vals, 1)
        baseline = np.polyval(coeffs, x_rt)
    elif method == 'quadratic':
        coeffs = np.polyfit(x_vals, y_vals, 2)
        baseline = np.polyval(coeffs, x_rt)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    return y - baseline, baseline


def detect_peaks(x_rt, y_corrected, n_peaks, manual_peaks=[], peaks_are_mw=True,
                 calibration_func=None, x_axis_type="MW"):
    """
    Return peak centers in RT, their heights, and the count.
    """
    if len(manual_peaks) == 0:
        indices, _ = find_peaks(y_corrected, distance=200, width=50)
        x_peaks_rt = x_rt[indices]
        y_peaks = y_corrected[indices]
        if len(y_peaks) >= n_peaks:
            top_indices = np.argsort(y_peaks)[-n_peaks:][::-1]
            x_peaks_rt = x_peaks_rt[top_indices]
            y_peaks = y_peaks[top_indices]
            n_peaks_found = n_peaks
        else:
            st.warning(f"Found only {len(y_peaks)} peaks, but {n_peaks} were expected.")
            n_peaks_found = len(y_peaks)
    else:
        x_peaks_rt, y_peaks = [], []
        for peak in manual_peaks:
            if peaks_are_mw and x_axis_type == "MW" and calibration_func:
                rt = calibration_func['mw_to_rt'](peak)
            else:
                rt = peak
            idx = np.argmin(np.abs(x_rt - rt))
            x_peaks_rt.append(x_rt[idx])
            y_peaks.append(y_corrected[idx])
        n_peaks_found = len(manual_peaks) if len(manual_peaks) < n_peaks else n_peaks

    return np.array(x_peaks_rt), np.array(y_peaks), n_peaks_found


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussians(x_rt, y_corrected, x_peaks_rt, y_peaks, n_peaks, peak_width_range):
    """
    Return best_fit (array of gaussians NxM), params list, and chosen width.
    """
    best_fit = None
    best_residual = np.inf
    best_width = peak_width_range[0]
    best_fit_params = []

    for width in range(peak_width_range[0], peak_width_range[1]):
        y_current = y_corrected.copy()
        gaussians = []
        params_list = []
        try:
            for i in range(n_peaks):
                mu = x_peaks_rt[i]
                idx = np.argmin(np.abs(x_rt - mu))
                start, end = max(0, idx - width), min(len(x_rt), idx + width)
                initial_guess = [y_peaks[i], mu, 0.1]
                params, _ = curve_fit(gaussian, x_rt[start:end], y_current[start:end], p0=initial_guess)
                y_fit = gaussian(x_rt, *params)
                gaussians.append(y_fit)
                params_list.append(params)
                y_current -= y_fit
            residual = np.sum(np.abs(y_current))
            if residual < best_residual:
                best_residual = residual
                best_fit = np.array(gaussians)
                best_fit_params = params_list
                best_width = width
        except (RuntimeError, IndexError):
            continue

    return best_fit, best_fit_params, best_width


def calculate_molecular_weight_averages(x_mw, y_signal, peak_ranges):
    """
    Compute Mn, Mw, Đ within each enabled range using positive signal only.
    """
    results = []
    for peak_name, ranges in peak_ranges.items():
        if not ranges.get("enabled", False):
            continue
        left, right = ranges["left"], ranges["right"]
        mask = (x_mw >= left) & (x_mw <= right)
        if np.sum(mask) == 0:
            continue
        x_region = x_mw[mask]
        y_region = np.maximum(y_signal[mask], 0)
        # Sort by ascending x to avoid sign issues
        order = np.argsort(x_region)
        x_region = x_region[order]
        y_region = y_region[order]
        # Averages
        # Mn = ∫y dx / ∫(y/x) dx
        num_mn = trapezoid(y_region, x_region)
        den_mn = trapezoid(y_region / x_region, x_region)
        mn = num_mn / den_mn if den_mn > 0 else 0.0
        # Mw = ∫(y x) dx / ∫y dx
        num_mw = trapezoid(y_region * x_region, x_region)
        den_mw = num_mn
        mw = num_mw / den_mw if den_mw > 0 else 0.0
        dispersity = (mw / mn) if mn > 0 else 0.0
        results.append({
            'Peak': peak_name,
            'Mn (g/mol)': int(mn) if mn > 0 else 0,
            'Mw (g/mol)': int(mw) if mw > 0 else 0,
            'Đ': f"{dispersity:.2f}",
        })
    return pd.DataFrame(results)


def create_plot(x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors,
                original_data_label, original_data_color, plot_sum, x_axis_type, x_lim, y_lim,
                font_family, font_size, fig_size, x_label, y_label, x_label_style,
                y_label_style, legend_style, integration_ranges=None):
    """
    Create final figure with optional shaded integration ranges.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Original data
    ax.plot(x_plot, y_corrected, label=original_data_label, linewidth=2, color=original_data_color, zorder=2)

    # Fitted peaks
    if best_fit is not None and len(best_fit) > 0:
        for i, (fit, _) in enumerate(zip(best_fit, area_percentages)):
            ax.plot(x_plot, fit, color=peak_colors[i], label=peak_names[i], zorder=2)

        if plot_sum:
            sum_gaussians = np.sum(best_fit, axis=0)
            ax.plot(x_plot, sum_gaussians, '--', color='black', linewidth=1.5, label='Sum of Gaussians', zorder=2)

    # Shade integration ranges only under the original curve and only within visible x-limits
    if integration_ranges:
        for peak_name, ranges in integration_ranges.items():
            if not ranges.get("enabled", False):
                continue
            left, right = float(ranges["left"]), float(ranges["right"])
            # Intersect with current plot x-limits
            left_vis = max(left, x_lim[0])
            right_vis = min(right, x_lim[1])
            if right_vis <= left_vis:
                continue
            mask = (x_plot >= left_vis) & (x_plot <= right_vis)
            if not np.any(mask):
                continue
            x_range = x_plot[mask]
            y_range = np.maximum(y_corrected[mask], 0.0)
            # Sort ascending to avoid polygon inversion
            order = np.argsort(x_range)
            x_range = x_range[order]
            y_range = y_range[order]
            # Boundaries
            ax.axvline(x=left, color='red', linestyle=':', alpha=0.7, linewidth=1)
            ax.axvline(x=right, color='red', linestyle=':', alpha=0.7, linewidth=1)
            # Fill baseline to curve (under only)
            ax.fill_between(x_range, 0.0, y_range, alpha=0.15, color='red', zorder=1)

    # Axis scales and limits
    if x_axis_type == "MW":
        ax.set_xscale('log')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Fonts with fallback
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if font_family not in available_fonts:
            fallback_fonts = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Arial", "Helvetica", "sans-serif"]
            for fallback in fallback_fonts:
                if fallback in available_fonts:
                    font_family = fallback
                    break
            else:
                font_family = available_fonts[0] if available_fonts else "sans-serif"

        font_prop_x = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in x_label_style else 'normal',
            weight='bold' if 'bold' in x_label_style else 'normal',
        )
        font_prop_y = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in y_label_style else 'normal',
            weight='bold' if 'bold' in y_label_style else 'normal',
        )
        font_prop_legend = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in legend_style else 'normal',
            weight='bold' if 'bold' in legend_style else 'normal',
        )
        ax.set_xlabel(x_label, fontproperties=font_prop_x)
        ax.set_ylabel(y_label, fontproperties=font_prop_y)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontproperties(fm.FontProperties(family=font_family, size=font_size))
        ax.legend(prop=font_prop_legend)
    except Exception as e:
        st.warning(f"Could not set custom font: {str(e)}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

    ax.grid(False)
    fig.tight_layout()
    return fig


def run_deconvolution(
        data_array,
        calib_array=None,
        x_axis_type="MW",
        x_lim=[1e3, 1e7],
        y_lim=[-0.02, 1.0],
        n_peaks=4,
        plot_sum=False,
        manual_peaks=[],
        peaks_are_mw=True,
        peak_names=["Peak 1", "Peak 2", "Peak 3", "Peak 4"],
        peak_colors=['#FFbf00', '#06d6a0', '#118ab2', '#073b4c'],
        peak_width_range=[100, 400],
        baseline_method='None',
        baseline_ranges=[],
        original_data_color='#ef476f',
        original_data_label='Original Data',
        font_family='Times New Roman',
        font_size=12,
        fig_size=(8, 5),
        x_label="Molecular weight (g/mol)",
        y_label="Normalized Response",
        x_label_style="normal",
        y_label_style="normal",
        legend_style="normal",
        integration_ranges=None
):
    """
    Orchestrate: prepare data, baseline-correct, fit Gaussians, compute areas, and return figure + tables.
    """
    plt.rcdefaults()

    # Raw arrays
    x_raw = data_array[:, 0].astype(float)
    y_raw = data_array[:, 1].astype(float)

    calibration_func = None

    if x_axis_type == "MW":
        if calib_array is None:
            raise ValueError("Calibration array is required for molecular weight plotting")

        retention_time_calib = calib_array[:, 0].astype(float)
        log_mw_calib = calib_array[:, 1].astype(float)
        f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')
        f_rt = interp1d(log_mw_calib, retention_time_calib, kind='linear', fill_value='extrapolate')
        calibration_func = {
            'mw_to_rt': lambda mw: f_rt(np.log10(mw)),
            'rt_to_mw': lambda rt: 10 ** f_log_mw(rt),
        }
        # Convert MW x-limits to RT limits (higher MW -> lower RT)
        rt_min = calibration_func['mw_to_rt'](x_lim[1])
        rt_max = calibration_func['mw_to_rt'](x_lim[0])
        rt_lim = [rt_min, rt_max]

        # Normalize y over visible RT window
        mask_range = (x_raw > rt_lim[0]) & (x_raw < rt_lim[1])
        max_y = np.max(y_raw[mask_range]) if np.any(mask_range) else 1.0
        y_raw = y_raw / max_y

        # Clip to visible RT window
        mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
        x_rt = x_raw[mask]
        y_formatted = y_raw[mask]

        # For plotting, convert RT to MW (descending x is possible)
        x_plot = calibration_func['rt_to_mw'](x_rt)
    else:
        # RT mode: use x-limits directly
        rt_lim = x_lim
        mask_range = (x_raw > rt_lim[0]) & (x_raw < rt_lim[1])
        max_y = np.max(y_raw[mask_range]) if np.any(mask_range) else 1.0
        y_raw = y_raw / max_y
        mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
        x_rt = x_raw[mask]
        y_formatted = y_raw[mask]
        x_plot = x_rt
        calibration_func = None

    # Baseline correction on the visible window only
    y_corrected, baseline = baseline_correction(x_rt, y_formatted, x_plot, baseline_method, baseline_ranges)

    # Detect peaks in RT domain
    x_peaks_rt, y_peaks, n_peaks_found = detect_peaks(
        x_rt, y_corrected, n_peaks, manual_peaks, peaks_are_mw, calibration_func, x_axis_type
    )

    # Fit Gaussians
    best_fit, best_fit_params, best_width = fit_gaussians(
        x_rt, y_corrected, x_peaks_rt, y_peaks, n_peaks_found, peak_width_range
    )

    # Gaussian areas and sorting by value (MW or RT)
    gaussian_results_df = pd.DataFrame(columns=['Peak', 'Value', 'Gaussian Area %'])
    integration_results_df = pd.DataFrame(columns=['Peak', 'Integration Area', 'Integration Area %'])
    mw_results_df = pd.DataFrame(columns=['Peak', 'Mn (g/mol)', 'Mw (g/mol)', 'Đ'])

    area_percentages = []
    peak_values = []

    if best_fit is not None and len(best_fit_params) > 0:
        mus = [params[1] for params in best_fit_params]
        if x_axis_type == "MW":
            peak_values = [calibration_func['rt_to_mw'](mu) for mu in mus]
            value_column = 'Mn (g/mol)'  # naming kept for table consistency
        else:
            peak_values = mus
            value_column = 'RT (min)'

        # Gaussian area percentages over visible RT domain
        area_integrals = [trapezoid(gauss, x_rt) for gauss in best_fit]
        total_area = sum(area_integrals) if len(area_integrals) else 0.0
        area_percentages = [(a / total_area) * 100 if total_area > 0 else 0.0 for a in area_integrals]

        # Sort descending by value
        sorted_indices = np.argsort(peak_values)[::-1]
        best_fit = best_fit[sorted_indices]
        best_fit_params = [best_fit_params[i] for i in sorted_indices]
        area_percentages = [area_percentages[i] for i in sorted_indices]
        peak_values = [peak_values[i] for i in sorted_indices]

        # Ensure names/colors length
        while len(peak_names) < len(best_fit):
            peak_names.append(f"Peak {len(peak_names) + 1}")
        peak_names = peak_names[:len(best_fit)]
        default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232',
                          '#a832a8', '#32a852', '#3264a8', '#a86432', '#6432a8']
        while len(peak_colors) < len(best_fit):
            peak_colors.append(default_colors[len(peak_colors) % len(default_colors)])
        peak_colors = peak_colors[:len(best_fit)]

        rows = []
        for name, value, pct in zip(peak_names, peak_values, area_percentages):
            if x_axis_type == "MW":
                rows.append({'Peak': name, value_column: int(value), 'Gaussian Area %': f"{pct:.1f}"})
            else:
                rows.append({'Peak': name, value_column: f"{value:.2f}", 'Gaussian Area %': f"{pct:.1f}"})
        gaussian_results_df = pd.DataFrame(rows)

    # Integration areas over visible RT range only; y clamped to >= 0; x_rt sorted ascending
    if integration_ranges:
        integration_areas = []
        for name in peak_names:
            if name in integration_ranges and integration_ranges[name].get("enabled", False):
                left = float(integration_ranges[name]["left"])
                right = float(integration_ranges[name]["right"])

                # Convert integration range boundaries to RT if in MW mode
                if x_axis_type == "MW" and calibration_func is not None:
                    # Convert MW boundaries to RT
                    left_rt = calibration_func['mw_to_rt'](right)  # Note: higher MW = lower RT
                    right_rt = calibration_func['mw_to_rt'](left)  # Note: lower MW = higher RT
                else:
                    left_rt = left
                    right_rt = right

                # Intersect with visible RT limits
                left_vis = max(left_rt, rt_lim[0])
                right_vis = min(right_rt, rt_lim[1])
                if right_vis <= left_vis:
                    integration_areas.append(0.0)
                    continue

                mask = (x_rt >= left_vis) & (x_rt <= right_vis)
                if np.sum(mask) > 0:
                    x_seg = x_rt[mask]
                    y_seg = np.maximum(y_corrected[mask], 0.0)
                    order = np.argsort(x_seg)
                    area = np.trapezoid(y_seg[order], x_seg[order])
                    integration_areas.append(max(area, 0.0))
                else:
                    integration_areas.append(0.0)
            else:
                integration_areas.append(0.0)

        # Percentages based on sum of enabled areas
        enabled_mask = [(name in integration_ranges and integration_ranges[name].get("enabled", False))
                        for name in peak_names]
        total_integration_area = sum([a for a, en in zip(integration_areas, enabled_mask) if en])
        integration_percentages = [(a / total_integration_area) * 100 if (en and total_integration_area > 0) else 0.0
                                   for a, en in zip(integration_areas, enabled_mask)]

        integration_rows = []
        for name, area, pct in zip(peak_names, integration_areas, integration_percentages):
            integration_rows.append({
                'Peak': name,
                'Integration Area': f"{area:.4f}",
                'Integration Area %': f"{pct:.1f}" if pct > 0 else "0.0",
            })
        integration_results_df = pd.DataFrame(integration_rows)

        # Molecular weight averages still use MW domain
        if x_axis_type == "MW" and calibration_func:
            mw_results_df = calculate_molecular_weight_averages(x_plot, y_corrected, integration_ranges)

    fig = create_plot(
        x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors,
        original_data_label, original_data_color, plot_sum, x_axis_type, x_lim, y_lim,
        font_family, font_size, fig_size, x_label, y_label, x_label_style,
        y_label_style, legend_style, integration_ranges
    )

    return fig, gaussian_results_df, integration_results_df, mw_results_df, x_plot, y_corrected, calibration_func