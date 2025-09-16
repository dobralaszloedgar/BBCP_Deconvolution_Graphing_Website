from Deconvolution import *
import streamlit as st
import numpy as np

# --------------------  Streamlit user interface  ----------------------
st.title("BBCP Deconvolution")

cal_file = st.file_uploader("Calibration curve (.txt)", type="txt", key="cal_file_uploader")
data_file = st.file_uploader("Chromatogram data (.txt)", type="txt", key="data_file_uploader")

with st.expander("Tunable parameters", key="tunable_params_expander"):
    col1, col2 = st.columns(2)
    with col1:
        mw_min = st.number_input("MW lower bound", 1e2, 1e8, 1e3, step=1e3, format="%e", key="mw_min_input")
        mw_max = st.number_input("MW upper bound", 1e3, 1e9, 1e7, step=1e6, format="%e", key="mw_max_input")
        y_low = st.number_input("Y-axis lower", -1.0, 0.99, -0.02, step=0.01, key="y_low_input")
        y_high = st.number_input("Y-axis upper", 0.1, 5.0, 1.0, step=0.1, key="y_high_input")
        peaks_n = st.slider("Number of peaks", 1, 6, 4, key="peaks_n_slider")
        plot_sum = st.checkbox("Plot sum of Gaussians", False, key="plot_sum_checkbox")
        original_data_color = st.color_picker("Original data color", value="#ef476f", key="original_data_color_picker")
        original_data_name = st.text_input("Original data label", value="Original Data", key="original_data_name_input")
    with col2:
        w_lo = st.number_input("Peak width search: start", 20, 800, 100, step=10, key="w_lo_input")
        w_hi = st.number_input("Peak width search: end", 50, 800, 450, step=10, key="w_hi_input")
        baseline_method = st.selectbox("Baseline method", ["flat", "linear", "quadratic"], index=2,
                                       key="baseline_method_select")
        bl_ranges_input = st.text_input("Baseline MW ranges (comma-sep pairs)",
                                        "1e3-1.2e3,14e3-21e3,9.5e6-1e7", key="bl_ranges_input")
        peaks_txt = st.text_input("Manual peaks (comma list, blank=auto)", "", key="peaks_txt_input")
        peaks_are_mw = st.checkbox("Manual peaks given as MW (unchecked=RT)", True, key="peaks_are_mw_checkbox")


# Parse baseline ranges string
def parse_ranges(txt):
    rngs = []
    for seg in txt.split(","):
        if "-" not in seg:
            continue
        lo, hi = map(float, seg.split("-"))
        rngs.append([lo, hi])
    return rngs


baseline_ranges = parse_ranges(bl_ranges_input)

# Peak names and colors customization
with st.expander("Peak names and colors (optional)", key="peak_customization_expander"):
    default_names = ["PS-b-2PLA-b-PS", "PS-b-2PLA", "PS-b", "PS", "Peak 5", "Peak 6"]
    default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232', '#a832a8']

    custom_names = []
    custom_colors = []

    for i in range(peaks_n):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Peak {i + 1} name",
                                 value=default_names[i] if i < len(default_names) else f"Peak {i + 1}",
                                 key=f"peak_name_{i}")
            custom_names.append(name)
        with col2:
            color = st.color_picker(f"Peak {i + 1} color",
                                    value=default_colors[i] if i < len(default_colors) else '#000000',
                                    key=f"peak_color_{i}")
            custom_colors.append(color)

if cal_file and data_file:
    try:
        calib = np.loadtxt(cal_file, delimiter="\t", skiprows=2)
        data = np.loadtxt(data_file, delimiter="\t", skiprows=2)

        manual_peaks = [float(p) for p in peaks_txt.split(",") if p]  # empty list==auto

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

        st.pyplot(fig)
        st.dataframe(table, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.info("Please ensure your files are in the correct format (tab-separated with 2 header rows)")
else:
    st.info("↑ Upload both calibration and data files to begin.")