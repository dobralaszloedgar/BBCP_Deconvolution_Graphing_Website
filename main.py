from Deconvolution import *
import streamlit as st
import numpy as np

# -------------------- Streamlit user interface ----------------------
st.title("BBCP Deconvolution")

# File uploaders
cal_file = st.file_uploader("Calibration curve (.txt)", type="txt")
data_file = st.file_uploader("Chromatogram data (.txt)", type="txt")

# Initialize session state for expanders
if 'expander_basic' not in st.session_state:
    st.session_state.expander_basic = True
if 'expander_advanced' not in st.session_state:
    st.session_state.expander_advanced = False

# Basic Parameters expander
with st.expander("Basic Parameters", expanded=st.session_state.expander_basic):
    col1, col2 = st.columns(2)
    with col1:
        mw_min = st.number_input("MW lower bound", 1e2, 1e8, 1e3, step=1e3, format="%e")
        mw_max = st.number_input("MW upper bound", 1e3, 1e9, 1e7, step=1e6, format="%e")
        y_low = st.number_input("Y-axis lower", -1.0, 0.99, -0.02, step=0.01)
        y_high = st.number_input("Y-axis upper", 0.1, 5.0, 1.0, step=0.1)

    with col2:
        peaks_n = st.slider("Number of peaks", 1, 6, 4)
        w_lo = st.number_input("Peak width search: start", 20, 800, 100, step=10)
        w_hi = st.number_input("Peak width search: end", 50, 800, 450, step=10)
        baseline_method = st.selectbox("Baseline method", ["flat", "linear", "quadratic"], index=2)

    # Baseline ranges
    bl_ranges_input = st.text_input("Baseline MW ranges (comma-sep pairs)",
                                    "1e3-1.2e3,14e3-21e3,9.5e6-1e7")

    # Manual peaks
    peaks_txt = st.text_input("Manual peaks (comma list, blank=auto)", "")
    peaks_are_mw = st.checkbox("Manual peaks given as MW (unchecked=RT)", True)

# Advanced Parameters expander
with st.expander("Peak Colors and Names", expanded=st.session_state.expander_advanced):
    # Appearance settings
    original_data_color = st.color_picker("Original data color", value="#ef476f")
    original_data_name = st.text_input("Original data label", value="Original Data")
    plot_sum = st.checkbox("Plot sum of Gaussians", False)

    # Peak names and colors
    st.write("Peak Names and Colors:")
    default_names = ["PS-b-2PLA-b-PS", "PS-b-2PLA", "PS-b", "PS", "Peak 5", "Peak 6"]
    default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232', '#a832a8']

    custom_names = []
    custom_colors = []

    for i in range(peaks_n):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Peak {i + 1} name",
                                 value=default_names[i] if i < len(default_names) else f"Peak {i + 1}",
                                 key=f"name_{i}")
            custom_names.append(name)
        with col2:
            color = st.color_picker(f"Peak {i + 1} color",
                                    value=default_colors[i] if i < len(default_colors) else '#000000',
                                    key=f"color_{i}")
            custom_colors.append(color)

# Parse baseline ranges string
def parse_ranges(txt):
    rngs = []
    for seg in txt.split(","):
        if "-" not in seg:
            continue
        try:
            lo, hi = map(float, seg.split("-"))
            rngs.append([lo, hi])
        except ValueError:
            st.warning(f"Invalid range format: {seg}. Skipping.")
    return rngs

# Process files if uploaded
if cal_file and data_file:
    try:
        # Parse baseline ranges
        baseline_ranges = parse_ranges(bl_ranges_input)

        # Load data (assuming tab-separated format with 2 header rows)
        calib = np.loadtxt(cal_file, delimiter="\t", skiprows=2)
        data = np.loadtxt(data_file, delimiter="\t", skiprows=2)

        # Parse manual peaks
        manual_peaks = []
        if peaks_txt.strip():
            for p in peaks_txt.split(","):
                try:
                    manual_peaks.append(float(p.strip()))
                except ValueError:
                    st.warning(f"Invalid peak value: {p}. Skipping.")

        # Run deconvolution
        fig, table = run_deconvolution(
            data_array=data,
            calib_array=calib,
            mw_lim=[mw_min, mw_max],
            y_lim=[y_low, y_high],
            n_peaks=peaks_n,
            plot_sum=plot_sum,
            manual_peaks=manual_peaks,
            peaks_are_mw=peaks_are_mw,
            peak_names=custom_names,
            peak_colors=custom_colors,
            peak_width_range=[int(w_lo), int(w_hi)],
            baseline_method=baseline_method,
            baseline_ranges=baseline_ranges,
            original_data_color=original_data_color,
            original_data_label=original_data_name
        )

        # Display results
        st.pyplot(fig, dpi=600)
        st.dataframe(table, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.info("Please ensure your files are in the correct format (tab-separated with 2 header rows)")
else:
    st.info("Upload both calibration and data files to begin.")