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
        y_signal: Signal intensity values
        peak_ranges: Dictionary containing integration ranges for each peak

    Returns:
        mw_results: DataFrame with Mn, Mw, and Đ for each peak
    """
    results = []

    for peak_name, ranges in peak_ranges.items():
        if ranges["enabled"]:
            left, right = ranges["left"], ranges["right"]

            # Find data points within the integration range
            mask = (x_mw >= left) & (x_mw <= right)
            if np.sum(mask) == 0:
                continue

            x_region = x_mw[mask]
            y_region = y_signal[mask]

            # Calculate molecular weight averages
            # Number average molecular weight (Mn)
            mn = np.trapz(y_region, x_region) / np.trapz(y_region / x_region, x_region)

            # Weight average molecular weight (Mw)
            mw = np.trapz(y_region * x_region, x_region) / np.trapz(y_region, x_region)

            # Dispersity (Đ)
            dispersity = mw / mn if mn > 0 else 0

            results.append({
                'Peak': peak_name,
                'Mn (g/mol)': int(mn),
                'Mw (g/mol)': int(mw),
                'Đ': f"{dispersity:.2f}"
            })

    return pd.DataFrame(results)


def create_plot(x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors,
                original_data_label, original_data_color, plot_sum, x_axis_type, x_lim, y_lim,
                font_family, font_size, fig_size, x_label, y_label, x_label_style,
                y_label_style, legend_style, integration_ranges=None):
    """
    Create the final plot with all data and formatting

    Args:
        x_plot: X-axis data for plotting
        y_corrected: Baseline-corrected response data
        best_fit: Fitted Gaussian curves
        area_percentages: Area percentages for each peak
        peak_names: Names for each peak
        peak_colors: Colors for each peak
        original_data_label: Label for original data
        original_data_color: Color for original data
        plot_sum: Whether to plot sum of Gaussians
        x_axis_type: Type of x-axis ("MW" or "RT")
        x_lim: X-axis limits
        y_lim: Y-axis limits
        font_family: Font family for text
        font_size: Font size
        fig_size: Figure size
        x_label: X-axis label
        y_label: Y-axis label
        x_label_style: X-axis label style
        y_label_style: Y-axis label style
        legend_style: Legend style
        integration_ranges: Integration ranges to mark on plot

    Returns:
        fig: Matplotlib figure object
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot original data
    ax.plot(x_plot, y_corrected, label=original_data_label, linewidth=2, color=original_data_color)

    # Plot fitted peaks
    if best_fit is not None and len(best_fit) > 0:
        for i, (fit, pct) in enumerate(zip(best_fit, area_percentages)):
            ax.plot(x_plot, fit, color=peak_colors[i], label=peak_names[i])

        if plot_sum:
            # Plot sum of Gaussians
            sum_gaussians = np.sum(best_fit, axis=0)
            ax.plot(x_plot, sum_gaussians, '--', color='black', linewidth=1.5, label='Sum of Gaussians')

    # Plot integration ranges if provided
    if integration_ranges:
        for peak_name, ranges in integration_ranges.items():
            if ranges["enabled"]:
                left, right = ranges["left"], ranges["right"]
                ax.axvline(x=left, color='red', linestyle=':', alpha=0.7, linewidth=1)
                ax.axvline(x=right, color='red', linestyle=':', alpha=0.7, linewidth=1)
                ax.axvspan(left, right, alpha=0.1, color='red')

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

        # Create font properties
        font_prop_x = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in x_label_style else 'normal',
            weight='bold' if 'bold' in x_label_style else 'normal'
        )

        font_prop_y = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in y_label_style else 'normal',
            weight='bold' if 'bold' in y_label_style else 'normal'
        )

        font_prop_legend = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='italic' if 'italic' in legend_style else 'normal',
            weight='bold' if 'bold' in legend_style else 'normal'
        )

        # Apply to labels
        ax.set_xlabel(x_label, fontproperties=font_prop_x)
        ax.set_ylabel(y_label, fontproperties=font_prop_y)

        # Apply to ticks
        font_prop_ticks = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='normal',
            weight='normal'
        )

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontproperties(font_prop_ticks)

        # Set font for legend
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

    Returns:
        fig: Matplotlib figure
        gaussian_results_df: DataFrame with Gaussian fitting results
        integration_results_df: DataFrame with integration results
        mw_results_df: DataFrame with molecular weight averages (MW mode only)
        x_plot: X-axis data for plotting
        y_corrected: Baseline-corrected y data
        calibration_func: Calibration functions for MW conversion
    """
    # Reset matplotlib to default settings
    plt.rcdefaults()

    # Extract and normalize data
    x_raw = data_array[:, 0].astype(float)
    y_raw = data_array[:, 1].astype(float)

    calibration_func = None

    if x_axis_type == "MW":
        # Molecular weight plotting requires calibration
        if calib_array is None:
            raise ValueError("Calibration array is required for molecular weight plotting")

        # Create interpolation functions for calibration data
        retention_time_calib = calib_array[:, 0].astype(float)
        log_mw_calib = calib_array[:, 1].astype(float)

        # Create interpolation functions for converting between RT and MW
        f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')
        f_rt = interp1d(log_mw_calib, retention_time_calib, kind='linear', fill_value='extrapolate')

        # Store calibration functions
        calibration_func = {
            'mw_to_rt': lambda mw: f_rt(np.log10(mw)),
            'rt_to_mw': lambda rt: 10 ** f_log_mw(rt)
        }

        # Convert MW limits to RT limits
        rt_min = calibration_func['mw_to_rt'](x_lim[1])  # Higher MW -> Lower RT
        rt_max = calibration_func['mw_to_rt'](x_lim[0])  # Lower MW -> Higher RT
        rt_lim = [rt_min, rt_max]

        # Find maximum y value within specified range and normalize
        mask_range = (x_raw > rt_lim[0]) & (x_raw < rt_lim[1])
        max_y = np.max(y_raw[mask_range]) if np.any(mask_range) else 1.0
        y_raw = y_raw / max_y

        # Filter data to specified retention time range
        mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
        x_rt = x_raw[mask]
        y_formatted = y_raw[mask]

        # Convert retention time to molecular weight for x-axis
        x_plot = calibration_func['rt_to_mw'](x_rt)

    else:
        # Retention time plotting
        rt_lim = x_lim

        # Find maximum y value within specified range and normalize
        mask_range = (x_raw > rt_lim[0]) & (x_raw < rt_lim[1])
        max_y = np.max(y_raw[mask_range]) if np.any(mask_range) else 1.0
        y_raw = y_raw / max_y

        # Filter data to specified retention time range
        mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
        x_rt = x_raw[mask]
        y_formatted = y_raw[mask]

        # Use retention time directly for x-axis
        x_plot = x_rt
        calibration_func = None

    # Apply baseline correction
    y_corrected, baseline = baseline_correction(x_rt, y_formatted, x_plot,
                                                baseline_method, baseline_ranges)

    # Detect peaks
    x_peaks_rt, y_peaks, n_peaks_found = detect_peaks(
        x_rt, y_corrected, n_peaks, manual_peaks, peaks_are_mw,
        calibration_func, x_axis_type
    )

    # Fit Gaussian functions
    best_fit, best_fit_params, best_width = fit_gaussians(
        x_rt, y_corrected, x_peaks_rt, y_peaks, n_peaks_found, peak_width_range
    )

    # Calculate Gaussian areas and percentages
    gaussian_results_df = pd.DataFrame(columns=['Peak', 'Value', 'Gaussian Area %'])
    integration_results_df = pd.DataFrame(columns=['Peak', 'Integration Area', 'Integration Area %'])
    mw_results_df = pd.DataFrame(columns=['Peak', 'Mn (g/mol)', 'Mw (g/mol)', 'Đ'])

    if best_fit is not None and len(best_fit_params) > 0:
        # Extract peak centers
        mus = [params[1] for params in best_fit_params]

        if x_axis_type == "MW":
            # Convert to molecular weights
            peak_values = [calibration_func['rt_to_mw'](mu) for mu in mus]
            value_column = 'Mn (g/mol)'
        else:
            # Use retention time directly
            peak_values = mus
            value_column = 'RT (min)'

        # Calculate Gaussian area percentages
        area_integrals = []
        for gaussian in best_fit:
            area_integral = trapezoid(gaussian, x_rt)
            area_integrals.append(area_integral)

        total_area = sum(area_integrals)
        area_percentages = [(a / total_area) * 100 for a in area_integrals]

        # Sort by value (highest to lowest)
        sorted_indices = np.argsort(peak_values)[::-1]
        best_fit = best_fit[sorted_indices]
        best_fit_params = [best_fit_params[i] for i in sorted_indices]
        area_percentages = [area_percentages[i] for i in sorted_indices]
        peak_values = [peak_values[i] for i in sorted_indices]

        # Ensure we have enough peak names and colors
        while len(peak_names) < len(best_fit):
            peak_names.append(f"Peak {len(peak_names) + 1}")
        peak_names = peak_names[:len(best_fit)]

        while len(peak_colors) < len(best_fit):
            default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232',
                              '#a832a8', '#32a852', '#3264a8', '#a86432', '#6432a8']
            peak_colors.append(default_colors[len(peak_colors) % len(default_colors)])
        peak_colors = peak_colors[:len(best_fit)]

        # Create Gaussian results table
        gaussian_data = []
        for i, (name, value, pct) in enumerate(zip(peak_names, peak_values, area_percentages)):
            if x_axis_type == "MW":
                gaussian_data.append({'Peak': name, value_column: int(value), 'Gaussian Area %': f"{pct:.1f}"})
            else:
                gaussian_data.append({'Peak': name, value_column: f"{value:.2f}", 'Gaussian Area %': f"{pct:.1f}"})

        gaussian_results_df = pd.DataFrame(gaussian_data)

        # Calculate integration results if ranges are provided
        if integration_ranges:
            integration_areas = []
            integration_percentages = []

            for peak_name in peak_names:
                if peak_name in integration_ranges and integration_ranges[peak_name]["enabled"]:
                    left = integration_ranges[peak_name]["left"]
                    right = integration_ranges[peak_name]["right"]

                    # Integrate the baseline-corrected data over the specified range
                    mask = (x_plot >= left) & (x_plot <= right)
                    if np.sum(mask) > 0:
                        area = np.trapz(y_corrected[mask], x_plot[mask])
                        integration_areas.append(area)
                    else:
                        integration_areas.append(0.0)
                else:
                    integration_areas.append(0.0)

            # Calculate percentages
            total_integration_area = sum(integration_areas)
            if total_integration_area > 0:
                integration_percentages = [(area / total_integration_area) * 100 for area in integration_areas]
            else:
                integration_percentages = [0.0] * len(integration_areas)

            # Create integration results table
            integration_data = []
            for i, (name, area, pct) in enumerate(zip(peak_names, integration_areas, integration_percentages)):
                integration_data.append({
                    'Peak': name,
                    'Integration Area': f"{area:.4f}",
                    'Integration Area %': f"{pct:.1f}" if pct > 0 else "0.0"
                })

            integration_results_df = pd.DataFrame(integration_data)

            # Calculate molecular weight averages if in MW mode and integration is enabled
            if x_axis_type == "MW" and calibration_func:
                mw_results_df = calculate_molecular_weight_averages(x_plot, y_corrected, integration_ranges)

    # Create the final plot
    fig = create_plot(
        x_plot, y_corrected, best_fit, area_percentages, peak_names, peak_colors,
        original_data_label, original_data_color, plot_sum, x_axis_type, x_lim, y_lim,
        font_family, font_size, fig_size, x_label, y_label, x_label_style,
        y_label_style, legend_style, integration_ranges
    )

    return fig, gaussian_results_df, integration_results_df, mw_results_df, x_plot, y_corrected, calibration_func