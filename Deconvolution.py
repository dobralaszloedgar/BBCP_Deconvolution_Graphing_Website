import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import pandas as pd
import streamlit as st
import matplotlib.font_manager as fm
import os

# Add pybaselines import
try:
    from pybaselines import Baseline

    PYBASELINES_AVAILABLE = True
except ImportError:
    PYBASELINES_AVAILABLE = False
    st.warning("pybaselines not installed. ARPLS baseline correction will not be available.")


def setup_custom_fonts():
    """Add custom fonts from the fonts directory to matplotlib's font manager"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(script_dir, 'fonts')

        # Check if fonts directory exists
        if os.path.exists(fonts_dir):
            # Add all fonts in the fonts directory
            font_files = fm.findSystemFonts(fontpaths=[fonts_dir])
            for font_file in font_files:
                fm.fontManager.addfont(font_file)

            # Clear the cache and update font list
            fm._load_fontmanager(try_read_cache=False)
            return True
        return False
    except Exception as e:
        st.warning(f"Could not set up custom fonts: {str(e)}")
        return False


# Call this function to set up custom fonts
setup_custom_fonts()


def baseline_correction(x_rt, y, x_plot, method='None', baseline_ranges=[]):
    """
    Apply baseline correction to the chromatogram data

    Args:
        x_rt: Retention time data
        y: Response data
        x_plot: X-axis data for plotting (MW or RT)
        method: Baseline correction method
        baseline_ranges: Ranges for baseline correction

    Returns:
        y_corrected: Baseline-corrected data
        baseline: Baseline that was subtracted
    """
    if method == 'None':
        # No baseline correction
        return y, np.zeros_like(y)
    elif method == 'arpls':
        if not PYBASELINES_AVAILABLE:
            raise ImportError("pybaselines is required for arpls baseline correction")
        baseline_fitter = Baseline()
        baseline = baseline_fitter.arpls(y, lam=1e12, tol=1e-4, max_iter=10)[0]
        return y - baseline, baseline

    ref_points = []
    required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}.get(method, 0)

    if len(baseline_ranges) != required_ranges:
        raise ValueError(f"{method} method requires {required_ranges} baseline ranges")

    # Calculate reference points from each baseline range
    for bl_range in baseline_ranges:
        # Find data points within the range
        mask = (x_plot >= bl_range[0]) & (x_plot <= bl_range[1])
        if np.sum(mask) == 0:
            raise ValueError(f"No data points in baseline range {bl_range}")

        # Calculate mean RT and response value within the range
        x_ref, y_ref = np.mean(x_rt[mask]), np.mean(y[mask])
        ref_points.append((x_ref, y_ref))

    # Extract x and y values from reference points
    x_vals = [p[0] for p in ref_points]
    y_vals = [p[1] for p in ref_points]

    # Calculate baseline based on selected method (in RT domain)
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

    # Return corrected data and baseline
    return y - baseline, baseline


def detect_peaks(x_rt, y_corrected, n_peaks, manual_peaks=[], peaks_are_mw=True,
                 calibration_func=None, x_axis_type="MW"):
    """
    Detect peaks in the chromatogram data

    Args:
        x_rt: Retention time data
        y_corrected: Baseline-corrected response data
        n_peaks: Number of peaks to detect
        manual_peaks: List of manually specified peaks
        peaks_are_mw: Whether manual peaks are in MW units
        calibration_func: Function to convert between RT and MW
        x_axis_type: Type of x-axis ("MW" or "RT")

    Returns:
        x_peaks_rt: Peak positions in retention time
        y_peaks: Peak heights
        n_peaks_found: Actual number of peaks found
    """
    if len(manual_peaks) == 0:
        # Automatic peak detection
        indices, _ = find_peaks(y_corrected, distance=200, width=50)
        x_peaks_rt = x_rt[indices]
        y_peaks = y_corrected[indices]

        # Select top peaks by height
        if len(y_peaks) >= n_peaks:
            top_indices = np.argsort(y_peaks)[-n_peaks:][::-1]
            x_peaks_rt = x_peaks_rt[top_indices]
            y_peaks = y_peaks[top_indices]
            n_peaks_found = n_peaks
        else:
            st.warning(f"Found only {len(y_peaks)} peaks, but {n_peaks} were expected.")
            n_peaks_found = len(y_peaks)  # Adjust to match what was found
    else:
        # Manual peak entry
        x_peaks_rt, y_peaks = [], []
        for peak in manual_peaks:
            if peaks_are_mw and x_axis_type == "MW" and calibration_func:
                # Convert from MW value to retention time
                rt = calibration_func['mw_to_rt'](peak)
            else:
                rt = peak  # Already in retention time or RT plotting

            # Find closest point in data
            idx = np.argmin(np.abs(x_rt - rt))
            x_peaks_rt.append(x_rt[idx])
            y_peaks.append(y_corrected[idx])

        # Adjust n_peaks if needed
        if len(manual_peaks) < n_peaks:
            st.warning(f"Only {len(manual_peaks)} peaks provided, but {n_peaks} expected.")
            n_peaks_found = len(manual_peaks)
        else:
            n_peaks_found = n_peaks

    return x_peaks_rt, y_peaks, n_peaks_found


def gaussian(x, amp, mu, sigma):
    """Gaussian function for peak fitting"""
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussians(x_rt, y_corrected, x_peaks_rt, y_peaks, n_peaks, peak_width_range):
    """
    Fit Gaussian functions to the detected peaks

    Args:
        x_rt: Retention time data
        y_corrected: Baseline-corrected response data
        x_peaks_rt: Peak positions in retention time
        y_peaks: Peak heights
        n_peaks: Number of peaks to fit
        peak_width_range: Range of window widths to try

    Returns:
        best_fit: Array of fitted Gaussian curves
        best_fit_params: Parameters for each Gaussian
        best_width: Best window width found
    """
    best_fit = None
    best_residual = np.inf
    best_width = peak_width_range[0]
    best_fit_params = []

    # Try different window widths to find best fit
    for width in range(peak_width_range[0], peak_width_range[1]):
        y_current = y_corrected.copy()
        gaussians = []
        params_list = []

        try:
            # Fit each peak with a Gaussian
            for i in range(n_peaks):
                mu = x_peaks_rt[i]
                idx = np.argmin(np.abs(x_rt - mu))
                start, end = max(0, idx - width), min(len(x_rt), idx + width)

                # Initial guess for Gaussian parameters [amplitude, mean, std dev]
                initial_guess = [y_peaks[i], mu, 0.1]
                params, _ = curve_fit(gaussian, x_rt[start:end], y_current[start:end], p0=initial_guess)

                # Calculate fitted Gaussian over full range
                y_fit = gaussian(x_rt, *params)
                gaussians.append(y_fit)
                params_list.append(params)

                # Subtract fitted Gaussian for next iteration
                y_current -= y_fit

            # Calculate residual to determine best fit
            residual = np.sum(np.abs(y_current))
            if residual < best_residual:
                best_residual = residual
                best_fit = np.array(gaussians)
                best_fit_params = params_list
                best_width = width
        except (RuntimeError, IndexError):
            # Skip if fitting fails
            continue

    return best_fit, best_fit_params, best_width


def calculate_molecular_weight_averages(x_mw, y_signal, peak_ranges):
    """
    Calculate molecular weight averages (Mn, Mw, Đ) for integration regions

    Args:
        x_mw: Molecular weight values
        y_signal: Signal intensity values (must be non-negative)
        peak_ranges: Dictionary containing integration ranges for each peak

    Returns:
        mw_results: DataFrame with Mn, Mw, and Đ for each peak
    """
    results = []
    y_signal_positive = np.maximum(0, y_signal)

    for peak_name, ranges in peak_ranges.items():
        if ranges["enabled"]:
            left, right = ranges["left"], ranges["right"]

            # Find data points within the integration range
            mask = (x_mw >= left) & (x_mw <= right)
            if np.sum(mask) < 2:  # Need at least 2 points for trapezoid rule
                continue

            x_region = x_mw[mask]
            y_region = y_signal_positive[mask]

            # Calculate molecular weight averages
            try:
                # Number average molecular weight (Mn)
                numerator_mn = np.trapz(y_region, x_region)
                denominator_mn = np.trapz(y_region / x_region, x_region)
                mn = numerator_mn / denominator_mn if denominator_mn > 0 else 0

                # Weight average molecular weight (Mw)
                numerator_mw = np.trapz(y_region * x_region, x_region)
                denominator_mw = np.trapz(y_region, x_region)
                mw = numerator_mw / denominator_mw if denominator_mw > 0 else 0

                # Dispersity (Đ)
                dispersity = mw / mn if mn > 0 else 0

                results.append({
                    'Peak': peak_name,
                    'Mn (g/mol)': int(mn),
                    'Mw (g/mol)': int(mw),
                    'Đ': f"{dispersity:.2f}"
                })
            except (ZeroDivisionError, ValueError):
                continue

    return pd.DataFrame(results)


def create_plot(x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors,
                original_data_label, original_data_color, plot_sum, x_axis_type, x_lim, y_lim,
                font_family, font_size, fig_size, x_label, y_label, x_label_style,
                y_label_style, legend_style, integration_ranges=None):
    """
    Create the final plot with all data and formatting
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot original data (baseline corrected)
    ax.plot(x_plot, y_corrected, label=original_data_label, linewidth=2, color=original_data_color)

    # Plot fitted peaks
    if best_fit is not None and len(best_fit) > 0:
        for i, fit in enumerate(best_fit):
            ax.plot(x_plot, fit, color=peak_colors[i], label=peak_names[i])

        if plot_sum:
            sum_gaussians = np.sum(best_fit, axis=0)
            ax.plot(x_plot, sum_gaussians, '--', color='black', linewidth=1.5, label='Sum of Gaussians')

    # Plot integration ranges if provided
    if integration_ranges:
        # Create an interpolation function for y_corrected vs x_plot
        # to get precise y-values at the integration boundaries.
        interp_y = interp1d(x_plot, y_corrected, bounds_error=False, fill_value=0)

        for peak_name, ranges in integration_ranges.items():
            if ranges["enabled"]:
                left, right = ranges["left"], ranges["right"]

                # Define the region for filling
                mask = (x_plot >= left) & (x_plot <= right)
                x_region = x_plot[mask]

                # Ensure we have points to fill
                if len(x_region) > 0:
                    y_region = y_corrected[mask]
                    # Clip y_region at 0 for filling to only show area above baseline
                    y_region_positive = np.maximum(0, y_region)
                    ax.fill_between(x_region, 0, y_region_positive, interpolate=True, color='red', alpha=0.15)

                # Get y-values at the exact boundaries for the dotted lines
                y_left = np.maximum(0, interp_y(left))
                y_right = np.maximum(0, interp_y(right))

                # Plot dotted lines from baseline (y=0) up to the curve
                ax.plot([left, left], [0, y_left], color='red', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.plot([right, right], [0, y_right], color='red', linestyle=':', alpha=0.7, linewidth=1.5)

    # Format plot
    if x_axis_type == "MW":
        ax.set_xscale('log')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Font handling with fallback
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        if font_family not in available_fonts:
            fallback_fonts = [
                "Times New Roman", "DejaVu Serif", "Liberation Serif",
                "Arial", "Helvetica", "sans-serif"
            ]
            for fallback in fallback_fonts:
                if fallback in available_fonts:
                    font_family = fallback
                    break
            else:
                font_family = available_fonts[0] if available_fonts else "sans-serif"

        font_prop_x = fm.FontProperties(family=font_family, size=font_size,
                                        style='italic' if 'italic' in x_label_style else 'normal',
                                        weight='bold' if 'bold' in x_label_style else 'normal')
        font_prop_y = fm.FontProperties(family=font_family, size=font_size,
                                        style='italic' if 'italic' in y_label_style else 'normal',
                                        weight='bold' if 'bold' in y_label_style else 'normal')
        font_prop_legend = fm.FontProperties(family=font_family, size=font_size,
                                             style='italic' if 'italic' in legend_style else 'normal',
                                             weight='bold' if 'bold' in legend_style else 'normal')

        ax.set_xlabel(x_label, fontproperties=font_prop_x)
        ax.set_ylabel(y_label, fontproperties=font_prop_y)

        font_prop_ticks = fm.FontProperties(family=font_family, size=font_size, style='normal', weight='normal')
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontproperties(font_prop_ticks)
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
        y_lim=[-0.02, 1],
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
    Main deconvolution function that orchestrates the entire process
    """
    # Reset matplotlib to default settings
    plt.rcdefaults()

    # Extract and normalize data
    x_raw, y_raw = data_array[:, 0].astype(float), data_array[:, 1].astype(float)

    calibration_func = None

    if x_axis_type == "MW":
        if calib_array is None:
            raise ValueError("Calibration array is required for molecular weight plotting")
        retention_time_calib, log_mw_calib = calib_array[:, 0].astype(float), calib_array[:, 1].astype(float)
        f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')
        f_rt = interp1d(log_mw_calib, retention_time_calib, kind='linear', fill_value='extrapolate')
        calibration_func = {'mw_to_rt': lambda mw: f_rt(np.log10(mw)), 'rt_to_mw': lambda rt: 10 ** f_log_mw(rt)}
        rt_min, rt_max = calibration_func['mw_to_rt'](x_lim[1]), calibration_func['mw_to_rt'](x_lim[0])
        rt_lim = [rt_min, rt_max]
    else:
        rt_lim = x_lim

    mask_range = (x_raw > rt_lim[0]) & (x_raw < rt_lim[1])
    max_y = np.max(y_raw[mask_range]) if np.any(mask_range) else 1.0
    y_raw = y_raw / max_y

    mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
    x_rt, y_formatted = x_raw[mask], y_raw[mask]

    x_plot = calibration_func['rt_to_mw'](x_rt) if x_axis_type == "MW" else x_rt

    y_corrected, baseline = baseline_correction(x_rt, y_formatted, x_plot, baseline_method, baseline_ranges)

    x_peaks_rt, y_peaks, n_peaks_found = detect_peaks(x_rt, y_corrected, n_peaks, manual_peaks, peaks_are_mw,
                                                      calibration_func, x_axis_type)

    best_fit, best_fit_params, best_width = fit_gaussians(x_rt, y_corrected, x_peaks_rt, y_peaks, n_peaks_found,
                                                          peak_width_range)

    gaussian_results_df = pd.DataFrame()
    integration_results_df = pd.DataFrame()
    mw_results_df = pd.DataFrame()
    area_percentages = []

    if best_fit is not None and len(best_fit_params) > 0:
        mus = [params[1] for params in best_fit_params]
        peak_values = [calibration_func['rt_to_mw'](mu) for mu in mus] if x_axis_type == "MW" else mus

        area_integrals = [trapezoid(gaussian, x_rt) for gaussian in best_fit]
        total_area = sum(area_integrals)
        area_percentages = [(a / total_area) * 100 if total_area > 0 else 0 for a in area_integrals]

        sorted_indices = np.argsort(peak_values)[::-1]
        best_fit = best_fit[sorted_indices]
        area_percentages = [area_percentages[i] for i in sorted_indices]
        peak_values = [peak_values[i] for i in sorted_indices]

        while len(peak_names) < len(best_fit): peak_names.append(f"Peak {len(peak_names) + 1}")
        peak_names = peak_names[:len(best_fit)]
        while len(peak_colors) < len(best_fit):
            default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232', '#a832a8', '#32a852', '#3264a8',
                              '#a86432', '#6432a8']
            peak_colors.append(default_colors[len(peak_colors) % len(default_colors)])
        peak_colors = peak_colors[:len(best_fit)]

        gaussian_data = []
        value_column = 'Mn (g/mol)' if x_axis_type == "MW" else 'RT (min)'
        for name, value, pct in zip(peak_names, peak_values, area_percentages):
            val_format = int(value) if x_axis_type == "MW" else f"{value:.2f}"
            gaussian_data.append({'Peak': name, value_column: val_format, 'Gaussian Area %': f"{pct:.1f}"})
        gaussian_results_df = pd.DataFrame(gaussian_data)

        if integration_ranges:
            integration_areas = []
            y_corrected_positive = np.maximum(0, y_corrected)

            for peak_name in peak_names:
                area = 0.0
                if peak_name in integration_ranges and integration_ranges[peak_name]["enabled"]:
                    left, right = integration_ranges[peak_name]["left"], integration_ranges[peak_name]["right"]
                    mask = (x_plot >= left) & (x_plot <= right)
                    if np.sum(mask) > 1:
                        area = np.trapezoid(y_corrected_positive[mask], x_plot[mask])
                integration_areas.append(area)

            total_integration_area = sum(integration_areas)
            integration_percentages = [(area / total_integration_area) * 100 if total_integration_area > 0 else 0 for area in integration_areas]

            integration_data = [{'Peak': name, 'Integration Area': f"{area:.4f}", 'Integration Area %': f"{pct:.1f}"}
                                for name, area, pct in zip(peak_names, integration_areas, integration_percentages)]
            integration_results_df = pd.DataFrame(integration_data)

            if x_axis_type == "MW" and calibration_func:
                mw_results_df = calculate_molecular_weight_averages(x_plot, y_corrected, integration_ranges)

    fig = create_plot(x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors, original_data_label,
                      original_data_color, plot_sum, x_axis_type, x_lim, y_lim, font_family, font_size, fig_size,
                      x_label, y_label, x_label_style, y_label_style, legend_style, integration_ranges)

    return fig, gaussian_results_df, integration_results_df, mw_results_df, x_plot, y_corrected, calibration_func
