from src.Deconvolution import run_deconvolution
import streamlit as st
import numpy as np
import requests
import tempfile
import os
import pandas as pd


def _clear_query_params_and_rerun():
    """Clear query parameters and rerun the app (for navigation)"""
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


def _round_sig(x: float, sig: int = 3) -> float:
    """Round to significant figures for clean default bounds."""
    try:
        if x == 0 or not np.isfinite(x):
            return x
        return float(f"{x:.{sig}g}")
    except Exception:
        return x


def _set_page_meta(title: str, icon: str):
    """
    Set page title and icon with fallback for when page_config is already set
    """
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Fallback: update title + favicon via JavaScript
        emoji = icon
        js = f"""
        <script>
        (function() {{
            const setTitle = (t) => {{ document.title = t; }};
            const setFavicon = (emoji) => {{
                const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>
                               <text x='50%' y='50%' dominant-baseline='central' text-anchor='middle' font-size='52'>{{emoji}}</text>
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


def download_default_file(url, filename):
    """Download default files from GitHub for example data"""
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


def load_array(path, skip_rows=2):
    # Auto-detect delimiter via csv.Sniffer (triggered by sep=None with engine='python')
    df = pd.read_csv(path, sep=None, engine='python', skiprows=skip_rows)
    return df.to_numpy(dtype=float)


def parse_ranges(inputs, is_mw=True):
    """
    Parse range inputs from text to numerical ranges

    Args:
        inputs: List of range strings (e.g., ["1e3-1.2e3"])
        is_mw: Whether the ranges are for molecular weight

    Returns:
        ranges: List of [left, right] numerical ranges
    """
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


def _setup_integration_sidebar_ui():
    """
    Renders the integration range controls in the sidebar if integration is
    enabled and peak data is available from a previous run.
    This function modifies st.session_state.peak_integration_ranges directly.
    """
    if not st.session_state.get('integration_enabled', False):
        # Clear integration tables when integration is disabled
        st.session_state.integration_table = None
        st.session_state.mw_table = None
        return

    # Check if we have peak names from a previous run to build the UI
    if 'gaussian_table' in st.session_state and st.session_state.gaussian_table is not None and not st.session_state.gaussian_table.empty:
        st.subheader("Integration Ranges")
        peak_names = st.session_state.gaussian_table['Peak'].tolist()
        x_plot = st.session_state.get('x_plot_data')

        if x_plot is not None:
            integration_ranges = st.session_state.get('peak_integration_ranges', {})
            current_peaks_n = st.session_state.get('peaks_n', 4)

            # Filter peak_names to only include the first n peaks based on current slider
            peak_names = peak_names[:current_peaks_n]

            x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
            x_axis_type = st.session_state.plot_x_axis

            for i, peak_name in enumerate(peak_names):
                # Only process peaks up to the current count
                if i >= current_peaks_n:
                    continue

                if peak_name not in integration_ranges:
                    # Set default range around the peak center if available
                    peak_row = st.session_state.gaussian_table[st.session_state.gaussian_table['Peak'] == peak_name]
                    if not peak_row.empty:
                        if x_axis_type == "MW":
                            peak_center = float(peak_row.iloc[0]['Mn (g/mol)'])
                            default_left = _round_sig(peak_center * 0.8, 3)
                            default_right = _round_sig(peak_center * 1.2, 3)
                        else:
                            peak_center = float(peak_row.iloc[0]['RT (min)'])
                            default_left = _round_sig(peak_center - 0.5, 3)
                            default_right = _round_sig(peak_center + 0.5, 3)
                    else:
                        default_left, default_right = _round_sig(x_min, 3), _round_sig(x_max, 3)

                    integration_ranges[peak_name] = {"enabled": True, "left": default_left, "right": default_right}

                with st.expander(f"Range for {peak_name}", expanded=False):
                    enabled = st.checkbox("Integrate", value=integration_ranges[peak_name].get("enabled", True),
                                          key=f"int_enabled_{peak_name}")

                    if enabled:
                        current_left = float(integration_ranges[peak_name].get("left", x_min))
                        current_right = float(integration_ranges[peak_name].get("right", x_max))

                        # Ensure left < right
                        if current_right < current_left:
                            current_right = current_left

                        # --- Number inputs for precise control ---
                        st.write("Precise bounds:")
                        left_col, right_col = st.columns(2)
                        with left_col:
                            final_left = st.number_input("Lower Bound",
                                                         min_value=float(x_min),
                                                         max_value=float(x_max),
                                                         value=float(current_left),
                                                         key=f"int_left_{peak_name}",
                                                         format="%e" if x_axis_type == "MW" else "%.3f",
                                                         step=100.0 if x_axis_type == "MW" else 0.01)
                        with right_col:
                            final_right = st.number_input("Upper Bound",
                                                          min_value=float(x_min),
                                                          max_value=float(x_max),
                                                          value=float(current_right),
                                                          key=f"int_right_{peak_name}",
                                                          format="%e" if x_axis_type == "MW" else "%.3f",
                                                          step=100.0 if x_axis_type == "MW" else 0.01)

                        # Ensure final_left < final_right
                        if final_right < final_left:
                            final_right = final_left

                        integration_ranges[peak_name] = {"enabled": True, "left": final_left, "right": final_right}

                    else:
                        integration_ranges[peak_name] = {"enabled": False, "left": x_min, "right": x_max}

            # Remove any integration ranges for peaks beyond current count
            current_peak_names = set(peak_names)
            for peak_name in list(integration_ranges.keys()):
                if peak_name not in current_peak_names:
                    del integration_ranges[peak_name]

            st.session_state.peak_integration_ranges = integration_ranges

    else:
        st.info("Run deconvolution once to define integration ranges.")


def setup_sidebar_ui():
    """
    Set up all the sidebar UI components

    Returns:
        params_dict: Dictionary containing all user parameters
        data_file: Uploaded or default data file
        cal_file: Uploaded or default calibration file
        cal_equation: Calibration equation parameters
    """

    st.header("Settings")

    # Data source selection
    data_source = st.radio("Select Data Source:", ["Use Example Data", "Upload My Own Data"], key="data_source")

    cal_file = None
    data_file = None
    cal_equation = None

    # File handling
    if data_source == "Use Example Data":
        st.info("Using example data to demonstrate the deconvolution process.")
        with st.spinner("Loading example data..."):
            DEFAULT_CAL_URL = "https://raw.githubusercontent.com/dobralaszloedgar/BBCP_Deconvolution_Graphing_Website/refs/heads/master/Calibration%20Curves/RI%20Calibration%20Curve%202024%20September.txt"
            DEFAULT_DATA_URL = "https://raw.githubusercontent.com/dobralaszloedgar/BBCP_Deconvolution_Graphing_Website/refs/heads/master/GPC%20Data/11.15.2024_GB_GRAFT_PS-b-2PLA.txt"

            cal_path = download_default_file(DEFAULT_CAL_URL, "default_cal.txt")
            data_path = download_default_file(DEFAULT_DATA_URL, "default_data.txt")

        if cal_path and data_path:
            cal_file = open(cal_path, 'r')
            data_file = open(data_path, 'r')
        else:
            st.stop()
    else:
        data_file = st.file_uploader("Chromatogram Data", type=["txt", "csv"], key="data_uploader", help="Upload a tab‑delimited text file (.txt or .csv).  \n First 2 rows are generally used for title and headers, so they will be removed.  \nPut Retention Time in column 1 and Intensity in column 2. Start your data on row 3.")
        if st.session_state.plot_x_axis == "MW":
            # Calibration source selection
            cal_source = st.radio("Calibration Source:", ["Upload Calibration File", "Enter Calibration Equation"],
                                  key="cal_source")

            if cal_source == "Upload Calibration File":
                cal_file = st.file_uploader("Calibration Curve", type=["txt", "csv"], key="cal_uploader", help="Upload a tab‑delimited calibration text file (.txt or .csv).  \nShould look the same as the data-file (same number of rows/columns), but only with calibration curve data. First 2 rows are generally used for title and headers, so they will be removed.  \nPut Retention Time in column 1 and Intensity in column 2. Start your data on row 3.")
            else:
                # Calibration equation input
                st.subheader("Calibration Equation")
                equation_type = st.selectbox("Equation Type", ["Linear", "Quadratic"], key="equation_type")

                if equation_type == "Linear":
                    st.latex(r"\log_{10}(MW) = a \cdot RT + b")
                    col1, col2 = st.columns(2)
                    with col1:
                        a = st.number_input("Slope (a)", value=-0.5, format="%.4f", key="linear_a")
                    with col2:
                        b = st.number_input("Intercept (b)", value=10.0, format="%.4f", key="linear_b")
                    cal_equation = {'type': 'linear', 'coefficients': [a, b]}

                else:  # Quadratic
                    st.latex(r"\log_{10}(MW) = a \cdot RT^2 + b \cdot RT + c")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        a = st.number_input("a coefficient", value=-0.1, format="%.4f", key="quad_a")
                    with col2:
                        b = st.number_input("b coefficient", value=2.0, format="%.4f", key="quad_b")
                    with col3:
                        c = st.number_input("c coefficient", value=5.0, format="%.4f", key="quad_c")
                    cal_equation = {'type': 'quadratic', 'coefficients': [a, b, c]}

        else:
            cal_file = None

        if (cal_file or cal_equation) and data_file:
            st.success("Data loaded successfully!")

    # X-axis type selection
    use_mw = st.toggle(
        "Retention Time ↔ Molecular Weight",
        value=(st.session_state.plot_x_axis == "MW"),
        help="Toggle between Molecular Weight and Retention Time for X-axis",
        key="x_axis_toggle"
    )

    # Update session state based on toggle
    new_toggle_state = "MW" if use_mw else "RT"
    if new_toggle_state != st.session_state.toggle_state:
        st.session_state.toggle_state = new_toggle_state
        st.session_state.plot_x_axis = new_toggle_state

        # Disable integration when switching to RT mode
        if new_toggle_state == "RT":
            st.session_state.integration_enabled = False

        # Update x_label when toggle changes - use session state callback
        if new_toggle_state == "MW":
            st.session_state.x_label = "Molecular weight (g/mol)"
        else:
            st.session_state.x_label = "Retention Time (min)"

        # Force a rerun to update the UI
        st.rerun()

    if st.session_state.plot_x_axis == "MW" and cal_file is None and cal_equation is None and data_source == "Upload My Own Data":
        st.warning("Calibration file or equation required for molecular weight plotting")

    # Number of Peaks
    peaks_n = st.slider("Number Of Peaks", 1, 10, 4, key="peaks_n")

    # Basic Parameters
    with st.expander("Basic Parameters", expanded=False):
        if st.session_state.plot_x_axis == "MW":
            mw_min = st.number_input("MW Lower Bound", 1e2, 1e8, 1e3, step=100.0, format="%e", key="mw_min")
            mw_max = st.number_input("MW Upper Bound", 1e3, 1e10, 1e7, step=10000.0, format="%e", key="mw_max")
        else:
            rt_min = st.number_input("RT Lower Bound (min)", 0.0, 100.0, 8.0, step=0.1, key="rt_min")
            rt_max = st.number_input("RT Upper Bound (min)", 0.0, 100.0, 19.0, step=0.1, key="rt_max")

        w_lo = st.number_input("Peak Width Search: Start", 20, 800, 100, step=10, key="w_lo")
        w_hi = st.number_input("Peak Width Search: End", 50, 800, 400, step=10, key="w_hi")
        y_low = st.number_input("Y-Axis Lower", -1.0, 0.99, -0.02, step=0.01, key="y_low")
        y_high = st.number_input("Y-Axis Upper", 0.1, 100.0, 1.05, step=0.01, key="y_high")

        # Manual peaks
        unit_label = "MW" if st.session_state.plot_x_axis == "MW" else "RT (min)"
        peaks_txt = st.text_input(f"Manual Peaks (comma list, blank=auto) in {unit_label}", "", key="peaks_txt")
        peaks_are_mw = st.checkbox(f"Manual Peaks Given As {unit_label}", True, key="peaks_are_mw")

    # Baseline Correction
    with st.expander("Baseline Correction", expanded=False):
        baseline_method = st.selectbox(
            "Baseline Correction Method",
            ["None", "arpls", "flat", "linear", "quadratic"],
            index=0,
            key="baseline_method"
        )

        # Baseline ranges UI
        baseline_ranges_inputs = []
        if baseline_method not in ["None", "arpls"]:
            unit = "MW" if st.session_state.plot_x_axis == "MW" else "RT (min)"
            required_ranges = {"flat": 1, "linear": 2, "quadratic": 3}.get(baseline_method, 0)
            st.write(f"Enter {required_ranges} baseline range(s) for {baseline_method} correction ({unit}):")
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

    # Peak Colors And Names
    with st.expander("Peak Appearance", expanded=False):
        default_names = ["Peak 1", "Peak 2", "Peak 3", "Peak 4", "Peak 5",
                         "Peak 6", "Peak 7", "Peak 8", "Peak 9", "Peak 10"]
        default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c', '#a83232',
                          '#a832a8', '#32a852', '#3264a8', '#a86432', '#6432a8']

        custom_names = []
        custom_colors = []
        peak_enabled = []

        original_data_name = st.text_input("Original Data Name", value="Original Data", key="original_data_name")
        original_data_color = st.color_picker("Original Data Color", value="#ef476f", key="original_data_color")

        for i in range(peaks_n):
            # Initialize peak enabled state if not exists
            peak_key = f"peak_enabled_{i}"
            if peak_key not in st.session_state:
                st.session_state[peak_key] = True

            enabled = st.checkbox(f"Enable Peak {i + 1}", value=st.session_state[peak_key], key=peak_key)
            peak_enabled.append(enabled)

            name = st.text_input(
                f"Peak {i + 1} Name",
                value=default_names[i] if i < len(default_names) else f"Peak {i + 1}",
                key=f"name_{i}"
            )
            custom_names.append(name)

            color = st.color_picker(
                f"Peak {i + 1} Color",
                value=default_colors[i] if i < len(default_colors) else '#000000',
                key=f"color_{i}"
            )
            custom_colors.append(color)

        plot_sum = st.checkbox("Plot Sum Of Gaussians", False, key="plot_sum")

    # Residual Peak Settings
    with st.expander("Residuals Plot", expanded=False):
        # Store previous sum state before changing it
        if "previous_plot_sum_state" not in st.session_state:
            st.session_state.previous_plot_sum_state = st.session_state.get("plot_sum", False)

        # Update plot_sum via a callback to avoid setting widget state during render
        def _sync_plot_sum_on_residual():
            if st.session_state.get("plot_residual", False):
                st.session_state["previous_plot_sum_state"] = False
                st.session_state["plot_sum"] = True
            else:
                if (
                        st.session_state.get("plot_sum", False)
                        and st.session_state.get("previous_plot_sum_state") is False
                ):
                    st.session_state["plot_sum"] = False

        plot_residual = st.checkbox(
            "Plot Residuals",
            False,
            key="plot_residual",
            on_change=_sync_plot_sum_on_residual,
        )
        residual_color = st.color_picker("Residuals Color", value="#7B3413", key="residual_color")

        if plot_residual:
            st.info("✓ Sum of Gaussians is automatically enabled when plotting residuals")

    # Peak Integration
    with st.expander("Peak Integration", expanded=False):
        # Disable integration in RT mode
        if st.session_state.plot_x_axis == "RT":
            st.session_state.integration_enabled = False
            integration_enabled = st.checkbox(
                "Enable Peak Integration",
                value=False,
                key="integration_enabled_checkbox",
                disabled=True,  # Disable the checkbox in RT mode
                help="Peak integration is only available in Molecular Weight mode"
            )
            st.info("⚠️ Peak integration is only available in Molecular Weight mode")
        else:
            integration_enabled = st.checkbox(
                "Enable Peak Integration",
                value=st.session_state.get('integration_enabled', False),
                key="integration_enabled_checkbox"
            )
            st.session_state.integration_enabled = integration_enabled

        # Call the integration ranges UI directly inside this expander
        if integration_enabled and st.session_state.plot_x_axis == "MW":
            _setup_integration_sidebar_ui()
        elif st.session_state.plot_x_axis == "RT":
            # Clear integration data when in RT mode
            st.session_state.integration_table = None
            st.session_state.mw_table = None
            st.session_state.peak_integration_ranges = {}

    # Appearance Settings
    with st.expander("Figure Appearance", expanded=False):
        common_fonts = sorted([
            "Arial", "Times New Roman", "Helvetica", "Verdana", "Georgia",
            "Courier New", "Tahoma", "Trebuchet MS", "Palatino", "Garamond",
            "Comic Sans MS", "Impact", "Lucida Console", "Lucida Sans Unicode",
            "Calibri", "Cambria", "Candara", "Segoe UI", "Optima", "Futura"
        ])
        default_font_index = common_fonts.index("Times New Roman") if "Times New Roman" in common_fonts else 0
        font_family = st.selectbox("Font Family", common_fonts, index=default_font_index, key="font_family")
        font_size = st.number_input("Font Size", 8, 20, 12, step=1, key="font_size")

        fig_width = st.number_input("Figure Width (inches)", 5.0, 15.0, 8.0, step=0.5, key="fig_width")
        fig_height = st.number_input("Figure Height (inches)", 4.0, 10.0, 5.0, step=0.5, key="fig_height")

        # Set default x-label based on current x-axis type
        if st.session_state.plot_x_axis == "MW":
            x_label_default = "Molecular weight (g/mol)"
        else:
            x_label_default = "Retention Time (min)"

        # Use session state for x_label if it exists, otherwise use default
        if 'x_label' not in st.session_state:
            st.session_state.x_label = x_label_default

        x_label = st.text_input("X-Axis Label", value=st.session_state.x_label, key="x_label")

        x_label_style = st.selectbox("X-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0,
                                     key="x_label_style")
        y_label = st.text_input("Y-Axis Label", "Normalized Response", key="y_label")
        y_label_style = st.selectbox("Y-Axis Label Style", ["normal", "italic", "bold", "bold italic"], index=0,
                                     key="y_label_style")
        legend_style = st.selectbox("Legend Style", ["normal", "italic", "bold", "bold italic"], index=0,
                                    key="legend_style")

    # Auto-update and manual update controls
    auto_update = st.checkbox("Auto-update graph", value=True,
                              help="Automatically update graph when parameters change",
                              key="auto_update")

    update_button = st.button("Update Graph",
                              help="Manually update the graph (useful when auto-update is disabled)",
                              key="update_button",
                              width='stretch')

    # Compile all parameters into a dictionary
    params_dict = {
        'data_source': data_source,
        'plot_x_axis': st.session_state.plot_x_axis,
        'mw_min': mw_min if st.session_state.plot_x_axis == "MW" else None,
        'mw_max': mw_max if st.session_state.plot_x_axis == "MW" else None,
        'rt_min': rt_min if st.session_state.plot_x_axis == "RT" else None,
        'rt_max': rt_max if st.session_state.plot_x_axis == "RT" else None,
        'y_low': y_low,
        'y_high': y_high,
        'peaks_n': peaks_n,
        'w_lo': w_lo,
        'w_hi': w_hi,
        'baseline_method': baseline_method,
        'baseline_ranges': baseline_ranges_inputs,
        'peaks_txt': peaks_txt,
        'peaks_are_mw': peaks_are_mw,
        'plot_sum': st.session_state.get('plot_sum', False),
        'plot_residual': plot_residual,
        'residual_color': residual_color,
        'custom_names': custom_names,
        'custom_colors': custom_colors,
        'peak_enabled': peak_enabled,
        'original_data_name': original_data_name,
        'original_data_color': original_data_color,
        'font_family': font_family,
        'font_size': font_size,
        'fig_width': fig_width,
        'fig_height': fig_height,
        'x_label': x_label,
        'y_label': y_label,
        'x_label_style': x_label_style,
        'y_label_style': y_label_style,
        'legend_style': legend_style,
        'integration_enabled': st.session_state.integration_enabled,
        'auto_update': auto_update
    }

    return params_dict, data_file, cal_file, cal_equation


def main():
    """Main function to run the Gaussian deconvolution app"""
    # Ensure tab title and icon reflect the Gaussian page
    _set_page_meta("Deconvolution", "📊")

    # Initialize session state variables
    if 'plot_x_axis' not in st.session_state:
        st.session_state.plot_x_axis = "MW"
    if 'last_fig' not in st.session_state:
        st.session_state.last_fig = None
    if 'gaussian_table' not in st.session_state:
        st.session_state.gaussian_table = None
    if 'integration_table' not in st.session_state:
        st.session_state.integration_table = None
    if 'mw_table' not in st.session_state:
        st.session_state.mw_table = None
    if 'residual_table' not in st.session_state:
        st.session_state.residual_table = None
    if 'graph_placeholder' not in st.session_state:
        st.session_state.graph_placeholder = None
    if 'table_placeholder' not in st.session_state:
        st.session_state.table_placeholder = None
    if 'toggle_state' not in st.session_state:
        st.session_state.toggle_state = "MW"
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {}
    if 'integration_enabled' not in st.session_state:
        st.session_state.integration_enabled = False
    if 'peak_integration_ranges' not in st.session_state:
        st.session_state.peak_integration_ranges = {}
    if 'last_integration_ranges' not in st.session_state:
        st.session_state.last_integration_ranges = {}
    if 'x_plot_data' not in st.session_state:
        st.session_state.x_plot_data = None
    if 'y_corrected_data' not in st.session_state:
        st.session_state.y_corrected_data = None
    if 'last_data_file' not in st.session_state:
        st.session_state.last_data_file = None
    if 'last_cal_file' not in st.session_state:
        st.session_state.last_cal_file = None
    if 'last_data_source' not in st.session_state:
        st.session_state.last_data_source = None
    if 'x_label' not in st.session_state:
        st.session_state.x_label = "Molecular weight (g/mol)"  # Default value
    if 'previous_plot_sum_state' not in st.session_state:
        st.session_state.previous_plot_sum_state = False

    # Back to launcher button at top
    if st.button("← Back to Launcher"):
        _clear_query_params_and_rerun()

    st.link_button("Help", f"https://github.com/dobralaszloedgar/BBCP_Deconvolution_Graphing_Website/blob/b0737a7566f7346ead3c4e68ea938ff175aa607c/README.md")

    # Main content area - only graph and table
    st.title("Gaussian Deconvolution")

    # SIDEBAR - All settings and parameters
    with st.sidebar:
        params_dict, data_file, cal_file, cal_equation = setup_sidebar_ui()

    # NEW: Check if data files have changed and clear previous results
    current_data_source = params_dict['data_source']

    # Safe way to get file names without causing errors
    current_data_file_name = None
    current_cal_file_name = None

    if data_file is not None:
        if hasattr(data_file, 'name'):
            current_data_file_name = data_file.name
        elif hasattr(data_file, 'getvalue'):  # For file-like objects
            current_data_file_name = "uploaded_data.txt"

    if cal_file is not None:
        if hasattr(cal_file, 'name'):
            current_cal_file_name = cal_file.name
        elif hasattr(cal_file, 'getvalue'):  # For file-like objects
            current_cal_file_name = "uploaded_cal.txt"

    # Check if data source or files have changed
    data_changed = (
            current_data_source != st.session_state.last_data_source or
            current_data_file_name != st.session_state.last_data_file or
            current_cal_file_name != st.session_state.last_cal_file
    )

    if data_changed and current_data_source == "Upload My Own Data":
        # Clear all previous results when new data is uploaded
        st.session_state.last_fig = None
        st.session_state.gaussian_table = None
        st.session_state.integration_table = None
        st.session_state.mw_table = None
        st.session_state.residual_table = None
        st.session_state.x_plot_data = None
        st.session_state.y_corrected_data = None
        st.session_state.peak_integration_ranges = {}

        # Clear the placeholders
        if st.session_state.graph_placeholder is not None:
            st.session_state.graph_placeholder.empty()
        if st.session_state.table_placeholder is not None:
            st.session_state.table_placeholder.empty()

        # Update the last file references
        st.session_state.last_data_source = current_data_source
        st.session_state.last_data_file = current_data_file_name
        st.session_state.last_cal_file = current_cal_file_name

    # MAIN CONTENT AREA - Only graph and table
    # Create placeholders for graph and table if they don't exist
    if st.session_state.graph_placeholder is None:
        st.session_state.graph_placeholder = st.empty()
    if st.session_state.table_placeholder is None:
        st.session_state.table_placeholder = st.empty()

    # Check for changes to trigger an update
    current_params = params_dict.copy()
    params_changed = current_params != st.session_state.get('last_params', {})

    current_integration_ranges = st.session_state.get('peak_integration_ranges', {})
    integration_ranges_changed = current_integration_ranges != st.session_state.get('last_integration_ranges', {})

    # Check if data was just loaded successfully
    data_just_loaded = False
    if data_file and (st.session_state.plot_x_axis == "RT" or cal_file or cal_equation):
        if st.session_state.last_data_file != current_data_file_name or \
                st.session_state.last_cal_file != current_cal_file_name or \
                st.session_state.last_data_source != current_data_source:
            data_just_loaded = True

    should_update = st.session_state.get('update_button', False) or \
                    (params_dict.get('auto_update', True) and (
                                params_changed or integration_ranges_changed or data_just_loaded))

    if should_update:
        # Store current params for comparison next time
        st.session_state.last_params = current_params
        st.session_state.last_integration_ranges = current_integration_ranges.copy()

        # Update graph and table
        if data_file and (st.session_state.plot_x_axis == "RT" or cal_file or cal_equation):
            try:
                is_mw = st.session_state.plot_x_axis == "MW"
                baseline_ranges = parse_ranges(params_dict['baseline_ranges'], is_mw) if params_dict[
                                                                                             'baseline_method'] not in [
                                                                                             "None", "arpls"] else []

                # Reset file pointers to beginning if they are file objects
                if hasattr(data_file, 'seek'):
                    data_file.seek(0)
                if cal_file and hasattr(cal_file, 'seek'):
                    cal_file.seek(0)

                data = load_array(data_file, skip_rows=2)
                calib = load_array(cal_file, skip_rows=2) if is_mw and cal_file else None

                manual_peaks = [float(p.strip()) for p in params_dict['peaks_txt'].split(",") if p.strip()]

                x_lim = [params_dict['mw_min'], params_dict['mw_max']] if is_mw else [params_dict['rt_min'],
                                                                                      params_dict['rt_max']]

                integration_ranges = st.session_state.peak_integration_ranges if params_dict[
                    'integration_enabled'] else None

                fig, gaussian_results_df, integration_results_df, mw_results_df, residual_results_df, x_plot, y_corrected, calibration_func = run_deconvolution(
                    data_array=data,
                    calib_array=calib,
                    calib_equation=cal_equation,
                    x_axis_type=st.session_state.plot_x_axis,
                    x_lim=x_lim,
                    y_lim=[params_dict['y_low'], params_dict['y_high']],
                    n_peaks=params_dict['peaks_n'],
                    plot_sum=params_dict['plot_sum'],
                    manual_peaks=manual_peaks,
                    peaks_are_mw=params_dict['peaks_are_mw'],
                    peak_names=params_dict['custom_names'],
                    peak_colors=params_dict['custom_colors'],
                    peak_enabled=params_dict['peak_enabled'],
                    peak_width_range=[int(params_dict['w_lo']), int(params_dict['w_hi'])],
                    baseline_method=params_dict['baseline_method'],
                    baseline_ranges=baseline_ranges,
                    original_data_color=params_dict['original_data_color'],
                    original_data_label=params_dict['original_data_name'],
                    font_family=params_dict['font_family'],
                    font_size=params_dict['font_size'],
                    fig_size=(params_dict['fig_width'], params_dict['fig_height']),
                    x_label=params_dict['x_label'],
                    y_label=params_dict['y_label'],
                    x_label_style=params_dict['x_label_style'],
                    y_label_style=params_dict['y_label_style'],
                    legend_style=params_dict['legend_style'],
                    integration_ranges=integration_ranges,
                    plot_residual=params_dict['plot_residual'],
                    residual_color=params_dict['residual_color'])

                st.session_state.last_fig, st.session_state.gaussian_table = fig, gaussian_results_df
                st.session_state.integration_table, st.session_state.mw_table = integration_results_df, mw_results_df
                st.session_state.residual_table = residual_results_df
                st.session_state.x_plot_data, st.session_state.y_corrected_data = x_plot, y_corrected
                st.session_state.last_data_source = current_data_source
                st.session_state.last_data_file = current_data_file_name
                st.session_state.last_cal_file = current_cal_file_name


            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
            finally:
                if params_dict['data_source'] == "Use Example Data":
                    for f in [cal_file, data_file]:
                        if f:
                            try:
                                f.close()
                                os.unlink(f.name)
                            except Exception:
                                pass
        elif params_dict['data_source'] == "Upload My Own Data":
            st.info("Upload your data and calibration files to begin.")

    # Display results
    if st.session_state.last_fig is not None:
        st.session_state.graph_placeholder.pyplot(st.session_state.last_fig, dpi=600, width='stretch')

        with st.session_state.table_placeholder.container():
            if st.session_state.gaussian_table is not None and not st.session_state.gaussian_table.empty:
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Gaussian Results", "Integration Results", "Molecular Weight Results", "Residual Results"])
                with tab1:
                    st.dataframe(st.session_state.gaussian_table, width='stretch')
                with tab2:
                    if (st.session_state.integration_table is not None and
                            not st.session_state.integration_table.empty and
                            st.session_state.integration_enabled):
                        st.dataframe(st.session_state.integration_table, width='stretch')
                    else:
                        st.info("Enable peak integration to see integration results.")
                with tab3:
                    if (st.session_state.mw_table is not None and
                            not st.session_state.mw_table.empty and
                            st.session_state.integration_enabled):
                        st.dataframe(st.session_state.mw_table, width='stretch')
                    else:
                        st.info("Molecular weight results are available in MW mode with integration enabled.")
                with tab4:
                    if (st.session_state.residual_table is not None and
                            not st.session_state.residual_table.empty and
                            params_dict['plot_residual']):
                        st.dataframe(st.session_state.residual_table, width='stretch')
                    else:
                        st.info("Enable residual peak plotting to see residual results.")


if __name__ == "__main__":
    main()
