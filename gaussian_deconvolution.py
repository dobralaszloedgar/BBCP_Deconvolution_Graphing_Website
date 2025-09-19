from Deconvolution import *
import streamlit as st
import numpy as np
import requests
import tempfile
import os
import time


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


def _set_page_meta(title: str, icon: str):
    """
    Try to set page config. If the launcher already called set_page_config,
    fall back to JS to update the tab title and favicon dynamically.
    """
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            initial_sidebar_state="collapsed",
        )
    except Exception:
        # Fallback: update title + favicon via a tiny script (works when page_config already set)
        emoji = icon
        js = f"""
        <script>
        (function() {{
            const setTitle = (t) => {{ document.title = t; }};
            const setFavicon = (emoji) => {{
                const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>
                               <text x='50%' y='50%' dominant-baseline='central' text-anchor='middle' font-size='52'>{emoji}</text>
                             </svg>`;
                const url = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
                let link = document.querySelector("link[rel='icon']") || document.createElement('link');
                link.setAttribute('rel', 'icon');
                link.setAttribute('href', url);
                document.head.appendChild(link);
            }};
            setTitle("{title}");
            setFavicon("{emoji}");
        }})();
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)


def main():
    # Ensure tab title and icon reflect the Gaussian page
    _set_page_meta("Deconvolution", "📊")

    # Initialize session state for graph update timing
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = 0
    if 'plot_x_axis' not in st.session_state:
        st.session_state.plot_x_axis = "MW"  # Default to molecular weight
    if 'last_input_time' not in st.session_state:
        st.session_state.last_input_time = 0
    if 'update_pending' not in st.session_state:
        st.session_state.update_pending = False
    if 'last_fig' not in st.session_state:
        st.session_state.last_fig = None
    if 'last_table' not in st.session_state:
        st.session_state.last_table = None
    if 'last_params_hash' not in st.session_state:
        st.session_state.last_params_hash = None
    if 'graph_placeholder' not in st.session_state:
        st.session_state.graph_placeholder = None
    if 'table_placeholder' not in st.session_state:
        st.session_state.table_placeholder = None

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
        col1, col2 = st.columns(2)
        with col1:
            data_file = st.file_uploader("Chromatogram Data (.txt)", type="txt")
        with col2:
            # Only show calibration upload if plotting against MW
            if st.session_state.plot_x_axis == "MW":
                cal_file = st.file_uploader("Calibration Curve (.txt)", type="txt")
            else:
                cal_file = None

    # X-axis type selection as toggle
    col1, col2 = st.columns([1, 3])
    with col1:
        # Toggle switch for X-axis selection
        use_mw = st.toggle(
            "Retention Time ↔ Molecular Weight",
            value=(st.session_state.plot_x_axis == "MW"),
            help="Toggle between Molecular Weight and Retention Time for X-axis"
        )

        # Update session state based on toggle
        if use_mw:
            st.session_state.plot_x_axis = "MW"
        else:
            st.session_state.plot_x_axis = "RT"

    with col2:
        if st.session_state.plot_x_axis == "MW" and cal_file is None and data_source == "Upload My Own Data":
            st.warning("Calibration file required for molecular weight plotting")

    # Initialize session state for expanders
    if 'expander_basic' not in st.session_state:
        st.session_state.expander_basic = False
    if 'expander_advanced' not in st.session_state:
        st.session_state.expander_advanced = False
    if 'expander_appearance' not in st.session_state:
        st.session_state.expander_appearance = False

    # Basic Parameters
    with st.expander("Basic Parameters", expanded=st.session_state.expander_basic):
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.plot_x_axis == "MW":
                mw_min = st.number_input("MW Lower Bound", 1e2, 1e8, 1e3, step=1000.0, format="%e")
                mw_max = st.number_input("MW Upper Bound", 1e3, 1e10, 1e7, step=1000000.0, format="%e")
            else:
                rt_min = st.number_input("RT Lower Bound (min)", 0.0, 100.0, 8.0, step=0.1)
                rt_max = st.number_input("RT Upper Bound (min)", 0.0, 100.0, 19.0, step=0.1)

            y_low = st.number_input("Y-Axis Lower", -1.0, 0.99, -0.02, step=0.01)
            y_high = st.number_input("Y-Axis Upper", 0.1, 100.0, 1.05, step=0.01)
        with col2:
            peaks_n = st.slider("Number Of Peaks", 1, 10, 4)
            w_lo = st.number_input("Peak Width Search: Start", 20, 800, 100, step=10)
            w_hi = st.number_input("Peak Width Search: End", 50, 800, 400, step=10)
            baseline_method = st.selectbox(
                "Baseline Correction Method",
                ["None", "arpls", "flat", "linear", "quadratic"],
                index=0
            )

            # Baseline ranges UI - only show for flat, linear, quadratic methods
            if baseline_method not in ["None", "arpls"]:
                unit = "MW" if st.session_state.plot_x_axis == "MW" else "RT (min)"
                required_ranges = {"flat": 1, "linear": 2, "quadratic": 3}.get(baseline_method, 0)
                st.write(f"Enter {required_ranges} baseline range(s) for {baseline_method} correction ({unit}):")
                baseline_ranges_inputs = []
                for i in range(required_ranges):
                    if st.session_state.plot_x_axis == "MW":
                        default_val = "1e3-1.2e3" if i == 0 else f"{i + 1}e4-{i + 2}e4" if i == 1 else f"{i + 1}e6-{i + 2}e6"
                    else:
                        default_val = f"10.0-11.0" if i == 0 else f"{12.0 + i}-{13.0 + i}" if i == 1 else f"{15.0 + i}-{16.0 + i}"
                    range_input = st.text_input(
                        f"Baseline Range {i + 1} ({unit})",
                        value=default_val, key=f"bl_range_{i}"
                    )
                    baseline_ranges_inputs.append(range_input)
            else:
                baseline_ranges_inputs = []

        # Manual peaks
        unit_label = "MW" if st.session_state.plot_x_axis == "MW" else "RT (min)"
        peaks_txt = st.text_input(f"Manual Peaks (comma list, blank=auto) in {unit_label}", "")
        peaks_are_mw = st.checkbox(f"Manual Peaks Given As {unit_label}", True)

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

            if st.session_state.plot_x_axis == "MW":
                x_label = st.text_input("X-Axis Label", "Molecular weight (g/mol)")
            else:
                x_label = st.text_input("X-Axis Label", "Retention Time (min)")

            x_label_style = st.selectbox("X-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0)
            y_label = st.text_input("Y-Axis Label", "Normalized Response")
            y_label_style = st.selectbox("Y-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0)
            legend_style = st.selectbox("Legend Style", ["normal", "italic", "bold", "bold italic"], index=0)

    # Debounce mechanism for automatic updates
    current_time = time.time()
    debounce_delay = 2.0  # 2 seconds debounce

    # Check if we should update the graph
    if current_time - st.session_state.last_input_time > debounce_delay and st.session_state.update_pending:
        st.session_state.update_pending = False
        st.session_state.last_update_time = current_time
        should_update = True
    else:
        should_update = False

    # Create a hash of current parameters to detect changes
    params_hash = hash((
        st.session_state.plot_x_axis,
        mw_min if st.session_state.plot_x_axis == "MW" else rt_min,
        mw_max if st.session_state.plot_x_axis == "MW" else rt_max,
        y_low, y_high, peaks_n, w_lo, w_hi, baseline_method,
        tuple(baseline_ranges_inputs), peaks_txt, peaks_are_mw,
        original_data_name, original_data_color,
        tuple(custom_names), tuple(custom_colors), plot_sum,
        font_family, font_size, fig_width, fig_height,
        x_label, y_label, x_label_style, y_label_style, legend_style
    ))

    # Mark input time when any parameter changes
    if params_hash != st.session_state.get('last_params_hash'):
        st.session_state.last_input_time = current_time
        st.session_state.update_pending = True
        st.session_state.last_params_hash = params_hash

    # Utilities
    def parse_ranges(inputs, is_mw=True):
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
                    # For single values, create a small range around the value
                    if is_mw:
                        rngs.append([val * 0.99, val * 1.01])
                    else:
                        rngs.append([val - 0.01, val + 0.01])
                except ValueError:
                    st.warning(f"Invalid value format: {inp}. Skipping.")
        return rngs

    # Process when data file is present (calibration file only needed for MW)
    if data_file and (st.session_state.plot_x_axis == "RT" or cal_file):
        # Create placeholders for graph and table if they don't exist
        if st.session_state.graph_placeholder is None:
            st.session_state.graph_placeholder = st.empty()
        if st.session_state.table_placeholder is None:
            st.session_state.table_placeholder = st.empty()

        # Display the last graph if available
        if st.session_state.last_fig is not None and st.session_state.last_table is not None:
            with st.session_state.graph_placeholder:
                st.pyplot(st.session_state.last_fig, dpi=600, width="content")
            with st.session_state.table_placeholder:
                st.dataframe(st.session_state.last_table, width="content")

        if should_update:
            try:
                is_mw = st.session_state.plot_x_axis == "MW"
                baseline_ranges = parse_ranges(baseline_ranges_inputs, is_mw) if baseline_method not in ["None",
                                                                                                         "arpls"] else []

                # Load data (assuming tab-separated format with 2 header rows)
                data = np.loadtxt(data_file, delimiter="\t", skiprows=2)

                # Load calibration if needed
                calib = None
                if is_mw and cal_file:
                    calib = np.loadtxt(cal_file, delimiter="\t", skiprows=2)
                elif is_mw:
                    st.error("Calibration file required for molecular weight plotting")
                    st.stop()

                # Manual peaks
                manual_peaks = []
                if peaks_txt.strip():
                    for p in peaks_txt.split(","):
                        try:
                            manual_peaks.append(float(p.strip()))
                        except ValueError:
                            st.warning(f"Invalid peak value: {p}. Skipping.")

                # Set limits based on x-axis type
                if is_mw:
                    x_lim = [mw_min, mw_max]
                else:
                    x_lim = [rt_min, rt_max]

                # Run deconvolution
                fig, table = run_deconvolution(
                    data_array=data,
                    calib_array=calib,
                    x_axis_type=st.session_state.plot_x_axis,
                    x_lim=x_lim,
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

                # Store the results
                st.session_state.last_fig = fig
                st.session_state.last_table = table

                # Update the display with the new graph and table
                with st.session_state.graph_placeholder:
                    st.pyplot(fig, dpi=600, width="content")
                with st.session_state.table_placeholder:
                    st.dataframe(table, width="content")

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.info("Please ensure your files are in the correct format (tab-separated with 2 header rows)")
            finally:
                # Clean up temporary files if example data was used
                if data_source == "Use Example Data":
                    try:
                        try:
                            if cal_file:
                                cal_file.close()
                        except Exception:
                            pass
                        try:
                            data_file.close()
                        except Exception:
                            pass
                        try:
                            if cal_file:
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
            if st.session_state.plot_x_axis == "MW":
                st.info("Upload both calibration and data files to begin.")
            else:
                st.info("Upload data file to begin.")


if __name__ == "__main__":
    main()