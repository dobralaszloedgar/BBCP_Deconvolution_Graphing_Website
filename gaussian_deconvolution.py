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

            # Set default font
            plt.rcParams['font.family'] = 'Arial'
    except Exception as e:
        # st.warning(f"Could not set up custom fonts: {e}")
        pass  # Silently fail if there's an issue with font setup


def gaussian(x, amplitude, mean, stddev):
    """1D Gaussian function."""
    # Ensure stddev is not zero to avoid division by zero
    if stddev == 0:
        return np.zeros_like(x)
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def multi_gaussian(x, *params):
    """Sum of multiple Gaussian functions."""
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, mean, stddev = params[i:i + 3]
        # Add constraint to keep stddev positive
        if stddev > 0:
            y += gaussian(x, amplitude, mean, np.abs(stddev))
    return y


def get_rt_from_mw(mw, a, b):
    """Power law calibration: RT = a * MW^b"""
    return a * (mw ** b)


def get_mw_from_rt(rt, a, b):
    """Inverse of power law calibration: MW = (RT / a)^(1/b)"""
    return (rt / a) ** (1 / b)


def fit_calibration(rt_values, mw_values):
    """Fit calibration curve and return the function."""
    if len(rt_values) < 2 or len(mw_values) < 2:
        return None, "At least two calibration points are required."

    try:
        # Fit a power law: RT = a * MW^b  =>  log(RT) = log(a) + b*log(MW)
        log_mw = np.log(mw_values)
        log_rt = np.log(rt_values)

        # Perform linear fit
        b, log_a = np.polyfit(log_mw, log_rt, 1)
        a = np.exp(log_a)

        # Calculate R^2 value
        predicted_log_rt = b * log_mw + log_a
        ss_res = np.sum((log_rt - predicted_log_rt) ** 2)
        ss_tot = np.sum((log_rt - np.mean(log_rt)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1

        # Return calibration function and its inverse
        return {
            "a": a,
            "b": b,
            "r_squared": r_squared,
            "rt_to_mw": lambda rt: get_mw_from_rt(rt, a, b),
            "mw_to_rt": lambda mw: get_rt_from_mw(mw, a, b),
        }, None
    except Exception as e:
        return None, f"Calibration failed: {e}"


def perform_deconvolution(x_data, y_data, peaks_rt, initial_stddev, initial_amplitude, bounds):
    """Performs Gaussian deconvolution."""
    num_peaks = len(peaks_rt)

    # Initial guesses: [amp1, mean1, std1, amp2, mean2, std2, ...]
    initial_guesses = []
    for i in range(num_peaks):
        initial_guesses.extend([initial_amplitude[i], peaks_rt[i], initial_stddev])

    # Fit the multi-Gaussian model
    try:
        popt, pcov = curve_fit(
            multi_gaussian,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=bounds,
            maxfev=5000  # Increased max iterations
        )
        return popt, None
    except RuntimeError as e:
        return None, f"Deconvolution fit failed. Try adjusting initial guesses or bounds. Error: {e}"


def calculate_mw_from_integration_regions(x_mw, y_signal, peak_ranges, baseline_y):
    """
    Calculates number-average (Mn), weight-average (Mw), and dispersity (Đ)
    for specified molecular weight integration regions.
    Args:
        x_mw: Molecular weight values
        y_signal: Signal intensity values (must be non-negative)
        peak_ranges: Dictionary containing integration ranges for each peak
        baseline_y: The calculated baseline values corresponding to y_signal
    Returns:
        mw_results: DataFrame with Mn, Mw, and Đ for each peak
    """
    results = []
    total_area = 0
    peak_areas = {}

    # First pass: calculate area of each peak
    for peak_name, (start_mw, end_mw) in peak_ranges.items():
        # Ensure start is less than end
        if start_mw >= end_mw:
            continue

        mask = (x_mw >= start_mw) & (x_mw <= end_mw)
        if not np.any(mask):
            peak_areas[peak_name] = 0
            continue

        x_peak = x_mw[mask]

        # Integrate signal above baseline, ensuring it's non-negative
        y_corrected = np.maximum(0, y_signal[mask] - baseline_y[mask])

        # Use trapezoidal rule for integration
        area = trapezoid(y_corrected, x_peak)
        peak_areas[peak_name] = area
        total_area += area

    # Second pass: calculate Mn, Mw, and PDI for each peak
    for peak_name, (start_mw, end_mw) in peak_ranges.items():
        if start_mw >= end_mw:
            continue

        mask = (x_mw >= start_mw) & (x_mw <= end_mw)
        if not np.any(mask):
            continue

        x_peak = x_mw[mask]

        # Signal is proportional to number of moles * MW (Ni * Mi)
        # For GPC, y_signal is proportional to wi, which is Ni * Mi
        ni = (y_signal[mask] / x_peak)

        # Baseline correction
        ni_baseline = (baseline_y[mask] / x_peak)
        ni_corrected = np.maximum(0, ni - ni_baseline)

        # Calculate sums for Mn and Mw
        sum_ni = trapezoid(ni_corrected, x_peak)
        sum_ni_mi = trapezoid(ni_corrected * x_peak, x_peak)
        sum_ni_mi2 = trapezoid(ni_corrected * x_peak ** 2, x_peak)

        if sum_ni > 0:
            mn = sum_ni_mi / sum_ni
            mw = sum_ni_mi2 / sum_ni_mi
            pdi = mw / mn if mn > 0 else 0

            area_percentage = (peak_areas[peak_name] / total_area) * 100 if total_area > 0 else 0

            results.append({
                "Peak": peak_name,
                "Mn": mn,
                "Mw": mw,
                "Đ (PDI)": pdi,
                "Area (%)": area_percentage
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


def run_deconvolution():
    st.title("GPC Deconvolution App")
    setup_custom_fonts()

    # --- Sidebar ---
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

        st.header("2. Axis Configuration")
        x_axis_type = st.radio("X-axis data represents:", ("Retention Time (min)", "Molecular Weight (Da)"),
                               key="x_axis_type_radio")
        x_col = st.text_input("Column name/index for X-axis", "0")
        y_col = st.text_input("Column name/index for Y-axis", "1")

        header_row = st.number_input("Header row (set to None if no header)", value=0, format="%d")
        if header_row < 0: header_row = None

        st.header("3. Baseline Correction")
        use_baseline_correction = st.checkbox("Enable Baseline Correction", True)
        baseline_method = "arpls"
        if use_baseline_correction and PYBASELINES_AVAILABLE:
            lam_param = st.slider("ARPLS Lambda (λ)", 1e2, 1e9, 1e6, format="%e")
            p_param = st.slider("ARPLS p", 0.001, 1.0, 0.01)

    if not uploaded_file:
        st.info("Please upload a data file to begin.")
        return

    # --- Data Loading ---
    try:
        df = pd.read_csv(uploaded_file, header=header_row, delim_whitespace=True)
        x_input = pd.to_numeric(df.iloc[:, int(x_col)]).values
        y_input = pd.to_numeric(df.iloc[:, int(y_col)]).values
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- Calibration Section ---
    calibration_func = None
    is_calibrated = False

    # Calibration is needed if input is RT and we want to see MW
    if x_axis_type == "Retention Time (min)":
        with st.expander("GPC Calibration (RT -> MW)", expanded=True):
            st.write("Enter pairs of Retention Time (min) and corresponding Molecular Weight (Da) standards.")

            if 'cal_points' not in st.session_state:
                st.session_state.cal_points = [{"rt": 10, "mw": 100000}, {"rt": 20, "mw": 1000}]

            def draw_cal_points():
                for i, point in enumerate(st.session_state.cal_points):
                    cols = st.columns([3, 3, 1])
                    st.session_state.cal_points[i]['rt'] = cols[0].number_input(f"RT {i + 1} (min)", value=point['rt'],
                                                                                key=f"rt_{i}")
                    st.session_state.cal_points[i]['mw'] = cols[1].number_input(f"MW {i + 1} (Da)", value=point['mw'],
                                                                                format="%d", key=f"mw_{i}")
                    if cols[2].button("❌", key=f"del_{i}"):
                        st.session_state.cal_points.pop(i)
                        st.rerun()

            draw_cal_points()

            if st.button("Add Calibration Point"):
                st.session_state.cal_points.append({"rt": 0, "mw": 0})
                st.rerun()

            rt_vals = [p['rt'] for p in st.session_state.cal_points if p['rt'] > 0 and p['mw'] > 0]
            mw_vals = [p['mw'] for p in st.session_state.cal_points if p['rt'] > 0 and p['mw'] > 0]

            if len(rt_vals) >= 2:
                calibration_func, error = fit_calibration(rt_vals, mw_vals)
                if error:
                    st.error(error)
                else:
                    st.success(f"Calibration successful! R² = {calibration_func['r_squared']:.4f}")
                    is_calibrated = True
                    x_mw = calibration_func['rt_to_mw'](x_input)
                    x_axis_label = "Molecular Weight (Da)"
            else:
                st.warning("Please provide at least two valid calibration points.")
                x_mw = None
                x_axis_label = "Retention Time (min)"
    else:  # Input is already MW
        is_calibrated = True
        x_mw = x_input
        x_axis_label = "Molecular Weight (Da)"

    # Use retention time if not calibrated, otherwise use MW
    x_plot = x_mw if is_calibrated and x_mw is not None else x_input
    y_plot = y_input

    # --- Baseline Correction ---
    baseline_y = np.zeros_like(y_plot)
    y_corrected = y_plot
    if use_baseline_correction and PYBASELINES_AVAILABLE:
        baseline_fitter = Baseline(x_data=x_plot)
        try:
            baseline_y, _ = baseline_fitter.arpls(y_plot, lam=lam_param, p=p_param)
            y_corrected = y_plot - baseline_y
        except Exception as e:
            st.error(f"Baseline correction failed: {e}")

    # --- Deconvolution & Integration Settings ---
    st.header("4. Deconvolution and Integration")

    # Peak finding vs Manual
    peak_source = st.radio("Peak Definition Method", ["Automated Peak Finding", "Manual Integration Ranges"], index=1)

    popt = None  # Deconvolution parameters

    if peak_source == "Automated Peak Finding":
        st.subheader("Automated Peak Finding Settings")
        height_threshold = st.slider("Peak Height Threshold", 0.0, 1.0, 0.1, 0.01)
        peak_prominence = st.slider("Peak Prominence", 0.0, 1.0, 0.1, 0.01)
        max_peaks = st.number_input("Maximum Number of Peaks", 1, 20, 5)

        # Find peaks on the baseline-corrected data
        peaks_indices, _ = find_peaks(y_corrected, height=height_threshold * np.max(y_corrected),
                                      prominence=peak_prominence)

        # Limit number of peaks
        if len(peaks_indices) > max_peaks:
            # Sort by prominence or height and take the top ones
            peak_heights = y_corrected[peaks_indices]
            sorted_indices = np.argsort(peak_heights)[::-1]
            peaks_indices = peaks_indices[sorted_indices[:max_peaks]]

        peaks_rt_found = x_plot[peaks_indices]
        st.write(f"Found {len(peaks_rt_found)} peaks.")

        st.subheader("Deconvolution Settings")
        initial_stddev_guess = st.slider("Initial StdDev Guess", 0.01, max(x_plot) / 4 if max(x_plot) > 0 else 1.0, 0.5)

        # Bounds for parameters [amp, mean, stddev]
        lower_bounds = []
        upper_bounds = []
        for peak_rt in peaks_rt_found:
            lower_bounds.extend([0, peak_rt - initial_stddev_guess * 2, 0.01])  # amp, mean, std
            upper_bounds.extend([np.inf, peak_rt + initial_stddev_guess * 2, initial_stddev_guess * 5])

        # Perform deconvolution
        popt, fit_error = perform_deconvolution(x_plot, y_corrected, peaks_rt_found, initial_stddev_guess,
                                                y_corrected[peaks_indices], (lower_bounds, upper_bounds))
        if fit_error:
            st.error(fit_error)

    # --- Manual Integration Ranges ---
    integration_ranges = {}
    if peak_source == "Manual Integration Ranges" and is_calibrated:
        st.subheader("Define Integration Ranges (MW)")
        num_ranges = st.number_input("Number of integration ranges", 1, 10, 1)

        min_mw_limit = float(np.min(x_mw))
        max_mw_limit = float(np.max(x_mw))

        # Use columns for a cleaner layout
        for i in range(num_ranges):
            st.markdown(f"---")
            st.markdown(f"**Range {i + 1}**")
            cols = st.columns([1, 1, 2])

            # Use log scale for sliders, but linear for number input
            log_min = np.log10(min_mw_limit)
            log_max = np.log10(max_mw_limit)

            # Default values for the slider
            default_start = np.log10(np.quantile(x_mw, 0.2 + i * 0.2)) if num_ranges > 1 else log_min
            default_end = np.log10(np.quantile(x_mw, 0.4 + i * 0.2)) if num_ranges > 1 else log_max

            # Session state to keep slider and number_input in sync
            state_key = f"range_{i}"
            if state_key not in st.session_state:
                st.session_state[state_key] = (10 ** default_start, 10 ** default_end)

            # Number inputs
            start_val = cols[0].number_input(f"Start MW {i + 1}", min_value=min_mw_limit, max_value=max_mw_limit,
                                             value=st.session_state[state_key][0], key=f"num_start_{i}")
            end_val = cols[1].number_input(f"End MW {i + 1}", min_value=min_mw_limit, max_value=max_mw_limit,
                                           value=st.session_state[state_key][1], key=f"num_end_{i}")

            # Logarithmic slider
            log_range = cols[2].slider(f"Adjust Range {i + 1}", log_min, log_max,
                                       (np.log10(start_val), np.log10(end_val)), key=f"slider_{i}")

            # Update state based on which widget was used last
            # A bit of a hack to sync widgets, Streamlit doesn't have a native callback for this
            if (10 ** log_range[0], 10 ** log_range[1]) != (start_val, end_val):
                # if slider was moved, update number inputs in next rerun
                st.session_state[state_key] = (10 ** log_range[0], 10 ** log_range[1])
                st.rerun()
            else:
                # if number input was changed, update state
                st.session_state[state_key] = (start_val, end_val)

            integration_ranges[f"Peak {i + 1}"] = (start_val, end_val)

    # --- Plotting ---
    st.header("5. Results")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original data
    ax.plot(x_plot, y_plot, label="Original Data", color='black')

    # Plot baseline
    if use_baseline_correction:
        ax.plot(x_plot, baseline_y, label="Baseline", color="gray", linestyle="--")

    # Plot deconvoluted peaks
    if popt is not None:
        fitted_curve = multi_gaussian(x_plot, *popt) + (baseline_y if use_baseline_correction else 0)
        ax.plot(x_plot, fitted_curve, label="Total Fit", color="red", linestyle="-", linewidth=2)
        for i in range(0, len(popt), 3):
            peak_curve = gaussian(x_plot, *popt[i:i + 3]) + (baseline_y if use_baseline_correction else 0)
            ax.plot(x_plot, peak_curve, linestyle="--", label=f"Peak {i // 3 + 1}")

    # Plot manual integration ranges
    if integration_ranges and is_calibrated:
        # Interpolate functions to get exact y-values at range boundaries
        interp_signal = interp1d(x_plot, y_plot, bounds_error=False, fill_value="extrapolate")
        interp_baseline = interp1d(x_plot, baseline_y, bounds_error=False, fill_value="extrapolate")

        colors = plt.cm.viridis(np.linspace(0, 1, len(integration_ranges)))
        for i, (name, (start_mw, end_mw)) in enumerate(integration_ranges.items()):
            if start_mw >= end_mw:
                continue

            # Fill area between curve and baseline
            mask = (x_plot >= start_mw) & (x_plot <= end_mw)
            ax.fill_between(x_plot[mask], baseline_y[mask], y_plot[mask], color=colors[i], alpha=0.4,
                            label=f"{name} Area")

            # Draw dotted vertical lines from baseline to curve
            y_start_curve = interp_signal(start_mw)
            y_start_base = interp_baseline(start_mw)
            ax.plot([start_mw, start_mw], [y_start_base, y_start_curve], 'r--', linewidth=1.5)

            y_end_curve = interp_signal(end_mw)
            y_end_base = interp_baseline(end_mw)
            ax.plot([end_mw, end_mw], [y_end_base, y_end_curve], 'r--', linewidth=1.5)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Signal Intensity")
    ax.set_title("GPC Chromatogram")

    if is_calibrated:  # Log scale for MW
        ax.set_xscale('log')

    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    st.pyplot(fig)

    # --- Results Table ---
    if integration_ranges and is_calibrated:
        st.subheader("Molecular Weight Averages from Integration")
        # Need to sort x_mw for interpolation if it's not already
        sort_indices = np.argsort(x_mw)
        x_mw_sorted = x_mw[sort_indices]
        y_plot_sorted = y_plot[sort_indices]
        baseline_y_sorted = baseline_y[sort_indices]

        mw_results_df = calculate_mw_from_integration_regions(x_mw_sorted, y_plot_sorted, integration_ranges,
                                                              baseline_y_sorted)

        if not mw_results_df.empty:
            st.dataframe(mw_results_df.style.format({
                "Mn": "{:,.0f}",
                "Mw": "{:,.0f}",
                "Đ (PDI)": "{:.3f}",
                "Area (%)": "{:.2f}%"
            }))
        else:
            st.warning("Could not calculate MW results. Check integration ranges.")
    elif popt is not None and is_calibrated:
        st.subheader("Deconvolution Results")
        # Here you would calculate Mn/Mw for each deconvoluted peak
        st.info("MW calculation for deconvoluted Gaussian peaks is not yet implemented.")


if __name__ == "__main__":
    run_deconvolution()
