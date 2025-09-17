from Deconvolution import *
import streamlit as st
import numpy as np
import requests
import tempfile
import os

def _clear_query_params_and_rerun():
    try:
        # New API
        st.query_params.clear()
    except Exception:
        # Old API: set to empty
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
    st.rerun()

def main():
    # Back to launcher
    if st.button("← Back to Launcher"):
        _clear_query_params_and_rerun()

    st.title("Gaussian Deconvolution")

    # Default file URLs
    DEFAULT_CAL_URL = "https://raw.githubusercontent.com/dobralaszloedgar/BBCP_Deconvolution_Graphing_Website/refs/heads/master/Calibration%20Curves/RI%20Calibration%20Curve%202024%20September.txt"
    DEFAULT_DATA_URL = "https://raw.githubusercontent.com/dobralaszloedgar/BBCP_Deconvolution_Graphing_Website/refs/heads/master/GPC%20Data/11.15.2024_GB_GRAFT_PS-b-2PLA.txt"

    # Function to download default files
    def download_default_file(url, filename):
        try:
            response = requests.get(url)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            st.error(f"Error downloading default file: {str(e)}")
            return None

    # Data source selection
    data_source = st.radio("Select Data Source:", ["Use Example Data", "Upload My Own Data"])

    cal_file = None
    data_file = None

    if data_source == "Use Example Data":
        st.info("Using example data to demonstrate the deconvolution process.")
        with st.spinner("Loading example data..."):
            cal_path = download_default_file(DEFAULT_CAL_URL, "default_cal.txt")
            data_path = download_default_file(DEFAULT_DATA_URL, "default_data.txt")
        if cal_path and data_path:
            cal_file = open(cal_path, 'r')
            data_file = open(data_path, 'r')
            st.success("Example data loaded successfully!")
            st.write("Calibration curve: RI Calibration Curve 2024 September.txt")
            st.write("Chromatogram data: 11.15.2024_GB_GRAFT_PS-b-2PLA.txt")
        else:
            st.stop()
    else:
        cal_file = st.file_uploader("Calibration Curve (.txt)", type="txt")
        data_file = st.file_uploader("Chromatogram Data (.txt)", type="txt")

    # Initialize session state for expanders
    if 'expander_basic' not in st.session_state:
        st.session_state.expander_basic = True
    if 'expander_advanced' not in st.session_state:
        st.session_state.expander_advanced = False
    if 'expander_appearance' not in st.session_state:
        st.session_state.expander_appearance = False

    # Basic Parameters
    with st.expander("Basic Parameters", expanded=st.session_state.expander_basic):
        col1, col2 = st.columns(2)
        with col1:
            mw_min = st.number_input("MW Lower Bound", 1e2, 1e8, 1e3, step=1e3, format="%e")
            mw_max = st.number_input("MW Upper Bound", 1e3, 1e9, 1e7, step=1e6, format="%e")
            y_low = st.number_input("Y-Axis Lower", -1.0, 0.99, -0.02, step=0.01)
            y_high = st.number_input("Y-Axis Upper", 0.1, 5.0, 1.0, step=0.1)
        with col2:
            peaks_n = st.slider("Number Of Peaks", 1, 10, 4)
            w_lo = st.number_input("Peak Width Search: Start", 20, 800, 100, step=10)
            w_hi = st.number_input("Peak Width Search: End", 50, 800, 400, step=10)
            baseline_method = st.selectbox(
                "Baseline Correction Method",
                ["None", "flat", "linear", "quadratic"],
                index=0
            )

            # Baseline ranges UI
            if baseline_method != "None":
                required_ranges = {"flat": 1, "linear": 2, "quadratic": 3}.get(baseline_method, 0)
                st.write(f"Enter {required_ranges} baseline range(s) for {baseline_method} correction:")
                baseline_ranges_inputs = []
                for i in range(required_ranges):
                    default_val = "1e3-1.2e3" if i == 0 else f"{i + 1}e4-{i + 2}e4" if i == 1 else f"{i + 1}e6-{i + 2}e6"
                    range_input = st.text_input(
                        f"Baseline Range {i + 1} (MW or MW-MW)",
                        value=default_val, key=f"bl_range_{i}"
                    )
                    baseline_ranges_inputs.append(range_input)
            else:
                baseline_ranges_inputs = []

        # Manual peaks
        peaks_txt = st.text_input("Manual Peaks (comma list, blank=auto)", "")
        peaks_are_mw = st.checkbox("Manual Peaks Given As MW (unchecked=RT)", True)

    # Peak Colors And Names
    with st.expander("Peak Colors And Names", expanded=st.session_state.expander_advanced):
        st.write("Peak Names And Colors:")
        default_names = ["Peak 1", "Peak 2", "Peak 3", "Peak 4", "Peak 5",
                         "Peak 6", "Peak 7", "Peak 8", "Peak 9", "Peak 10"]
        default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232',
                          '#a832a8', '#32a852', '#3264a8', '#a86432', '#6432a8']

        custom_names = []
        custom_colors = []

        cu1, cu2 = st.columns(2)
        with cu1:
            original_data_name = st.text_input("Original Data Name", value="Original Data")
        with cu2:
            original_data_color = st.color_picker("Original Data Color", value="#ef476f")

        for i in range(peaks_n):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input(
                    f"Peak {i + 1} Name",
                    value=default_names[i] if i < len(default_names) else f"Peak {i + 1}",
                    key=f"name_{i}"
                )
                custom_names.append(name)
            with col2:
                color = st.color_picker(
                    f"Peak {i + 1} Color",
                    value=default_colors[i] if i < len(default_colors) else '#000000',
                    key=f"color_{i}"
                )
                custom_colors.append(color)

        plot_sum = st.checkbox("Plot Sum Of Gaussians", False)

    # Appearance Settings
    with st.expander("Appearance Settings", expanded=st.session_state.expander_appearance):
        col1, col2 = st.columns(2)
        with col1:
            common_fonts = sorted([
                "Arial", "Times New Roman", "Helvetica", "Verdana", "Georgia",
                "Courier New", "Tahoma", "Trebuchet MS", "Palatino", "Garamond",
                "Comic Sans MS", "Impact", "Lucida Console", "Lucida Sans Unicode",
                "Calibri", "Cambria", "Candara", "Segoe UI", "Optima", "Futura"
            ])
            default_font_index = common_fonts.index("Times New Roman") if "Times New Roman" in common_fonts else 0
            font_family = st.selectbox("Font Family", common_fonts, index=default_font_index)
            font_size = st.number_input("Font Size", 8, 20, 12, step=1)
        with col2:
            fig_width = st.number_input("Figure Width (inches)", 5.0, 15.0, 8.0, step=0.5)
            fig_height = st.number_input("Figure Height (inches)", 4.0, 10.0, 5.0, step=0.5)
            x_label = st.text_input("X-Axis Label", "Molecular weight (g/mol)")
            x_label_style = st.selectbox("X-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0)
            y_label = st.text_input("Y-Axis Label", "Normalized Response")
            y_label_style = st.selectbox("Y-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0)
            legend_style = st.selectbox("Legend Style", ["normal", "italic", "bold", "bold italic"], index=0)

    # Utilities
    def parse_ranges(inputs):
        rngs = []
        for inp in inputs:
            if not inp.strip():
                continue
            if "-" in inp:
                try:
                    lo, hi = map(float, inp.split("-"))
                    rngs.append([lo, hi])
                except ValueError:
                    st.warning(f"Invalid range format: {inp}. Skipping.")
            else:
                try:
                    val = float(inp)
                    rngs.append([val * 0.99, val * 1.01])
                except ValueError:
                    st.warning(f"Invalid value format: {inp}. Skipping.")
        return rngs

    # Process when both files present
    if cal_file and data_file:
        try:
            baseline_ranges = parse_ranges(baseline_ranges_inputs) if baseline_method != "None" else []

            # Load data (assuming tab-separated format with 2 header rows)
            calib = np.loadtxt(cal_file, delimiter="\t", skiprows=2)
            data = np.loadtxt(data_file, delimiter="\t", skiprows=2)

            # Manual peaks
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
                original_data_label=original_data_name,
                font_family=font_family,
                font_size=font_size,
                fig_size=(fig_width, fig_height),
                x_label=x_label,
                y_label=y_label,
                x_label_style=x_label_style,
                y_label_style=y_label_style,
                legend_style=legend_style
            )

            # Display results
            st.pyplot(fig, dpi=600)
            st.dataframe(table, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.info("Please ensure your files are in the correct format (tab-separated with 2 header rows)")

        finally:
            # Clean up temporary files if example data was used
            if data_source == "Use Example Data":
                try:
                    try:
                        cal_file.close()
                    except Exception:
                        pass
                    try:
                        data_file.close()
                    except Exception:
                        pass
                    try:
                        os.unlink(cal_file.name)
                    except Exception:
                        pass
                    try:
                        os.unlink(data_file.name)
                    except Exception:
                        pass
                except Exception:
                    pass
    else:
        if data_source == "Upload My Own Data":
            st.info("Upload both calibration and data files to begin.")

if __name__ == "__main__":
    main()
