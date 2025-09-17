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


def run_deconvolution(
        data_array,
        calib_array,
        mw_lim=[1e3, 1e7],
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
        legend_style="normal"
):
    # Reset matplotlib to default settings to ensure clean state
    plt.rcdefaults()

    # Create interpolation functions for calibration data
    retention_time_calib = calib_array[:, 0].astype(float)
    log_mw_calib = calib_array[:, 1].astype(float)

    # Create interpolation functions for converting between RT and MW
    f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')
    f_rt = interp1d(log_mw_calib, retention_time_calib, kind='linear', fill_value='extrapolate')

    # Convert molecular weight to retention time
    def mw_to_rt(mw_value):
        log_mw = np.log10(mw_value)
        return f_rt(log_mw)

    # Convert molecular weight limits to retention time limits
    rt_min = mw_to_rt(mw_lim[1])  # Higher MW -> Lower RT
    rt_max = mw_to_rt(mw_lim[0])  # Lower MW -> Higher RT
    rt_lim = [rt_min, rt_max]

    # Find maximum y value within a specified x range
    def max_of_y_within_range(x_array, y_array, x_min, x_max):
        mask = (x_array > x_min) & (x_array < x_max)
        return np.max(y_array[mask]) if np.any(mask) else 1.0

    # Extract and normalize data
    x_raw = data_array[:, 0].astype(float)
    y_raw = data_array[:, 1].astype(float)
    max_y = max_of_y_within_range(x_raw, y_raw, rt_lim[0], rt_lim[1])
    y_raw = y_raw / max_y

    # Filter data to specified retention time range
    mask = (x_raw >= rt_lim[0]) & (x_raw <= rt_lim[1])
    x_rt = x_raw[mask]
    y_formatted = y_raw[mask]

    # Convert retention time to molecular weight for x-axis
    x_mw = 10 ** f_log_mw(x_rt)

    # Baseline correction function
    def baseline_correction(x_rt, y, x_mw, method='None'):
        if method == 'None':
            # No baseline correction
            return y, np.zeros_like(y)

        ref_points = []
        required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}.get(method, 0)

        if len(baseline_ranges) != required_ranges:
            raise ValueError(f"{method} method requires {required_ranges} baseline MW ranges")

        # Calculate reference points from each baseline range using MW
        for bl_range in baseline_ranges:
            # Find data points within the molecular weight range
            mask = (x_mw >= bl_range[0]) & (x_mw <= bl_range[1])
            if np.sum(mask) == 0:
                raise ValueError(f"No data points in baseline MW range {bl_range}")

            # Calculate mean RT and response value within the MW range
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

    # Apply baseline correction
    y_corrected, baseline = baseline_correction(x_rt, y_formatted, x_mw, method=baseline_method)

    # Peak Detection
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
        else:
            st.warning(f"Found only {len(y_peaks)} peaks, but {n_peaks} were expected.")
            n_peaks = len(y_peaks)  # Adjust to match what was found
    else:
        # Manual peak entry
        x_peaks_rt, y_peaks = [], []
        for peak in manual_peaks:
            if peaks_are_mw:
                # Convert from Mn value to retention time
                rt = mw_to_rt(peak)
            else:
                rt = peak  # Already in retention time

            # Find closest point in data
            idx = np.argmin(np.abs(x_rt - rt))
            x_peaks_rt.append(x_rt[idx])
            y_peaks.append(y_corrected[idx])

        # Adjust n_peaks if needed
        if len(manual_peaks) < n_peaks:
            st.warning(f"Only {len(manual_peaks)} peaks provided, but {n_peaks} expected.")
            n_peaks = len(manual_peaks)

    # Gaussian function for peak fitting
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Peak Fitting with Gaussian functions
    best_fit = None
    best_residual = np.inf
    best_width = peak_width_range[0]
    best_fit_params = []  # Store Gaussian parameters

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

    # Calculate molecular weight percentages and sort peaks by MW
    if best_fit is not None and len(best_fit_params) > 0:
        # Extract peak centers and calculate molecular weights
        mus = [params[1] for params in best_fit_params]
        mw_values = 10 ** f_log_mw(mus)  # Convert to molecular weights

        # Calculate area percentages in retention time domain
        area_integrals = []
        for gaussian in best_fit:
            # Simple area integration with respect to retention time
            area_integral = trapezoid(gaussian, x_rt)
            area_integrals.append(area_integral)

        # Calculate percentages based on areas
        total_area = sum(area_integrals)
        area_percentages = [(a / total_area) * 100 for a in area_integrals]

        # Sort by molecular weight (highest to lowest)
        sorted_indices = np.argsort(mw_values)[::-1]
        best_fit = best_fit[sorted_indices]
        best_fit_params = [best_fit_params[i] for i in sorted_indices]
        area_percentages = [area_percentages[i] for i in sorted_indices]
        mw_values = [mw_values[i] for i in sorted_indices]

        # Ensure we have enough peak names and colors
        while len(peak_names) < len(best_fit):
            peak_names.append(f"Peak {len(peak_names) + 1}")
        peak_names = peak_names[:len(best_fit)]

        while len(peak_colors) < len(best_fit):
            # Add default colors if not enough provided
            default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232',
                              '#a832a8', '#32a852', '#3264a8', '#a86432', '#6432a8']
            peak_colors.append(default_colors[len(peak_colors) % len(default_colors)])
        peak_colors = peak_colors[:len(best_fit)]

    # Create the plot
    fig, ax = plt.subplots(figsize=tuple(element / 1.5 for element in fig_size))

    # Plot original data
    ax.plot(x_mw, y_corrected, label=original_data_label, linewidth=2, color=original_data_color)

    # Plot fitted peaks
    if best_fit is not None and len(best_fit) > 0:
        for i, (fit, pct) in enumerate(zip(best_fit, area_percentages)):
            # Only show peak name in legend, not percentage
            ax.plot(x_mw, fit, color=peak_colors[i],
                    label=peak_names[i])  # Removed the percentage from label

        if plot_sum:
            # Plot sum of Gaussians
            sum_gaussians = np.sum(best_fit, axis=0)
            ax.plot(x_mw, sum_gaussians, '--', color='black', linewidth=1.5, label='Sum of Gaussians')

    # Format plot
    ax.set_xscale('log')
    ax.set_xlim(mw_lim)
    ax.set_ylim(y_lim)

    # Font handling with fallback
    try:
        # Check if the selected font exists
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # Try to find a suitable fallback if the selected font isn't available
        if font_family not in available_fonts:
            # List of fallback fonts to try in order of preference
            fallback_fonts = [
                "Times New Roman", "DejaVu Serif", "Liberation Serif",
                "Arial", "Helvetica", "sans-serif"
            ]

            for fallback in fallback_fonts:
                if fallback in available_fonts:
                    font_family = fallback
                    break
            else:
                # If no fallback found, use the first available font
                font_family = available_fonts[0] if available_fonts else "sans-serif"

        # Create separate font properties for x-label, y-label, and legend
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

        # Apply to labels with their respective styles
        ax.set_xlabel(x_label, fontproperties=font_prop_x)
        ax.set_ylabel(y_label, fontproperties=font_prop_y)

        # Apply to ticks and other text elements (use normal style for ticks)
        font_prop_ticks = fm.FontProperties(
            family=font_family,
            size=font_size,
            style='normal',
            weight='normal'
        )

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontproperties(font_prop_ticks)

        # Set font for legend
        if ax.get_legend():
            legend = ax.legend(prop=font_prop_legend)
        else:
            # Create legend if it doesn't exist
            ax.legend(prop=font_prop_legend)

    except Exception as e:
        st.warning(f"Could not set custom font: {str(e)}")
        # Fallback to default font settings
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if ax.get_legend():
            ax.legend()


    ax.grid(False)
    fig.tight_layout()

    # Create results table
    if best_fit is not None and len(best_fit) > 0:
        results_df = pd.DataFrame({
            'Peak': peak_names,
            'Mn (g/mol)': [int(mw) for mw in mw_values],
            'Area %': [round(pct, 1) for pct in area_percentages]
        })
    else:
        results_df = pd.DataFrame(columns=['Peak', 'Mn (g/mol)', 'Area %'])

    return fig, results_df