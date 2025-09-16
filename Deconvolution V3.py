import matplotlib.pyplot as plt
import streamlit as sl
import numpy as np
from fontTools.misc.cython import returns
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import matplotlib.font_manager

# Website
sl.title("BBCP Deconvolution V3")

cal_file = sl.file_uploader("Upload Calibration File (.txt)")
data_file = sl.file_uploader("Upload Data File (.txt)")

if cal_file and data_file:
    txt_file = data_file
    RI_calibration = cal_file

# Configuration
txt_file = data_file
#txt_file = r"G:\Edgar Dobra\GPC Samples\2024 Fall\11.15.2024_GB_GRAFT_PS-b-2PLA.txt"
mw_lim = [1e3, 1e7]  # Molecular weight limits for analysis (g/mol)
y_lim = [-0.02, 1]
number_of_peaks = 4
plot_sum_of_fitted_peaks = False  # Plots the sum of the fitted peaks if True, otherwise set to False
peaks = []  # For manual entry - can be Mn values
peaks_are_mw = True  # Set True if peaks are entered as Mn values, False for retention times
peak_names = ["PS-b-2PLA-b-PS", "PS-b-2PLA", "PS-b", "PS"]
peak_wideness_range = [100, 450]  # set to [100, 800] for default
baseline_method = 'quadratic'  # can change to 'flat' (1 range), 'linear'(2 ranges), or 'quadratic' (3 ranges)
baseline_ranges = [[1e3, 1.2e3], [14e3, 21e3], [9.5e6, 1e7]]  # MW ranges for baseline calculation

# Calibration and MW conversion
RI_calibration = cal_file
#RI_calibration = r"G:/Edgar Dobra\GPC Samples\Calibration Curves\RI Calibration Curve 2024 September.txt"
mw_x_lim = mw_lim  # Molecular weight limits for plotting

# Font settings
matplotlib.rcParams['font.family'] = 'Avenir Next LT Pro'
matplotlib.rcParams['font.size'] = 18

txt_file = r'{}'.format(txt_file)
RI_calibration = r'{}'.format(RI_calibration)

# Data Loading
data_array = np.loadtxt(txt_file, delimiter='\t', skiprows=2)

# Load calibration data
data_array_RI = np.loadtxt(RI_calibration, delimiter='\t', skiprows=2)
retention_time_calib = data_array_RI[:, 0].astype(float)
log_mw_calib = data_array_RI[:, 1].astype(float)

# Create interpolation functions for both directions
f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')
f_rt = interp1d(log_mw_calib, retention_time_calib, kind='linear', fill_value='extrapolate')


# Function to convert molecular weight to retention time
def mw_to_rt(mw_value):
    """Convert molecular weight to retention time using calibration curve"""
    log_mw = np.log10(mw_value)
    return f_rt(log_mw)


# Convert molecular weight limits to retention time limits
# Note: In GPC, higher MW corresponds to lower retention time
rt_min = mw_to_rt(mw_lim[1])  # Higher MW -> Lower RT
rt_max = mw_to_rt(mw_lim[0])  # Lower MW -> Higher RT
rt_lim = [rt_min, rt_max]

print(f"Analyzing MW range: {mw_lim[0]:.1e} to {mw_lim[1]:.1e} g/mol")
print(f"Corresponding RT range: {rt_lim[0]:.2f} to {rt_lim[1]:.2f} min")


# Find maximum y value within a specified x range
def max_of_y_within_range(x_array, y_array, x_min, x_max):
    """Find the maximum y value within a specified x range"""
    mask = (x_array > x_min) & (x_array < x_max)
    return np.max(y_array[mask]) if np.any(mask) else 1.0


# Extract and normalize data from the input file
def extract_data(data_index=0):
    """Extract data from file and normalize by maximum value in RT range"""
    x = data_array[:, data_index * 2]
    y = data_array[:, data_index * 2 + 1]
    x = x.astype(float)
    y = y.astype(float)
    max_y = max_of_y_within_range(x, y, rt_lim[0], rt_lim[1])
    return x, y / max_y


# Filter data to specified retention time range (derived from MW limits)
def format_data(x, y):
    """Restrict data to the specified retention time range"""
    mask = (x >= rt_lim[0]) & (x <= rt_lim[1])
    return x[mask], y[mask]


# Perform baseline correction using molecular weight ranges
def baseline_correction(x_rt, y, x_mw, method='linear'):
    """
    Correct baseline using specified method based on MW ranges

    Parameters:
    x_rt - retention time values
    y - y-axis values
    x_mw - molecular weight values corresponding to x_rt
    method - baseline correction method ('flat', 'linear', or 'quadratic')
    """
    ref_points = []
    required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}.get(method, 1)

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


# Gaussian function for peak fitting
def gaussian(x, amp, mu, sigma):
    """Gaussian function for peak fitting"""
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Main Processing
x_raw, y_raw = extract_data(0)
x_rt, y_formatted = format_data(x_raw, y_raw)

# Convert retention time to molecular weight for baseline correction and plotting
x_mw = 10 ** f_log_mw(x_rt)

# Apply baseline correction using molecular weight ranges
y_corrected, baseline = baseline_correction(x_rt, y_formatted, x_mw, method=baseline_method)

# Peak Detection
if len(peaks) == 0:
    # Automatic peak detection
    indices, _ = find_peaks(y_corrected, distance=200, width=50)
    x_peaks_rt = x_rt[indices]
    y_peaks = y_corrected[indices]

    # Select top peaks by height
    if len(y_peaks) >= number_of_peaks:
        top_indices = np.argsort(y_peaks)[-number_of_peaks:][::-1]
        x_peaks_rt = x_peaks_rt[top_indices]
        y_peaks = y_peaks[top_indices]
    else:
        print(f"Warning: Found only {len(y_peaks)} peaks, but {number_of_peaks} were expected.")
        print(f"Proceeding with available {len(y_peaks)} peaks.")
        number_of_peaks = len(y_peaks)  # Adjust to match what was found
else:
    # Manual peak entry - can be Mn values or retention times
    x_peaks_rt, y_peaks = [], []
    for peak in peaks:
        if peaks_are_mw:
            # Convert from Mn value to retention time
            rt = mw_to_rt(peak)
        else:
            rt = peak  # Already in retention time

        # Find closest point in data
        idx = np.argmin(np.abs(x_rt - rt))
        x_peaks_rt.append(x_rt[idx])
        y_peaks.append(y_corrected[idx])

    # Adjust number_of_peaks if needed
    if len(peaks) < number_of_peaks:
        print(f"Warning: Only {len(peaks)} peaks provided, but {number_of_peaks} expected.")
        number_of_peaks = len(peaks)

# Peak Fitting with Gaussian functions
best_fit = None
best_residual = np.inf
best_width = peak_wideness_range[0]
best_fit_params = []  # Store Gaussian parameters

# Try different window widths to find best fit
for width in range(peak_wideness_range[0], peak_wideness_range[1]):
    y_current = y_corrected.copy()
    gaussians = []
    params_list = []

    try:
        # Fit each peak with a Gaussian
        for i in range(number_of_peaks):
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

    # CHANGED SECTION: Calculate area percentages in retention time domain
    # Instead of using MW domain integration with Jacobian transformation
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

    # Ensure we have enough peak names
    while len(peak_names) < len(best_fit):
        peak_names.append(f"Peak {len(peak_names) + 1}")

    # Trim peak_names if we have fewer peaks than names
    peak_names = peak_names[:len(best_fit)]

# Visualization
plt.figure(figsize=(8, 5))
plt.subplots_adjust(bottom=0.19, left=0.19)

# Plot original data
plt.plot(x_mw, y_corrected, label='Original Data', linewidth=2, color='#ef476f')

# Plot fitted peaks - update to use area_percentages
if best_fit is not None and len(best_fit) > 0:
    colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c']
    for i, (fit, pct) in enumerate(zip(best_fit, area_percentages)):
        plt.plot(x_mw, fit, color=colors[i % len(colors)],
                 label=f'{peak_names[i]}: {pct:.1f}%')

    if plot_sum_of_fitted_peaks:
        # Plot sum of Gaussians
        sum_gaussians = np.sum(best_fit, axis=0)
        plt.plot(x_mw, sum_gaussians, '--', color='black', linewidth=1.5, label='Sum of Gaussians')


# Format plot
plt.xscale('log')
plt.xlim(mw_x_lim)
plt.ylim(y_lim)
plt.xlabel("Molecular weight (g/mol)", fontstyle='italic', fontweight='demi')
plt.ylabel("Normalized Response", fontstyle='italic', fontweight='demi')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("Figure1.png", dpi=1200)
plt.show()
