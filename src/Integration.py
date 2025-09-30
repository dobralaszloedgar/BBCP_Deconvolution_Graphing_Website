import numpy as np
import pandas as pd
import streamlit as st


def integrate_peak_region(x_data, y_data, left_bound, right_bound):
    """
    Integrate a region of the chromatogram between left and right bounds
    Only integrates positive values (above baseline)

    Args:
        x_data: X-axis data (MW or RT)
        y_data: Y-axis data (response)
        left_bound: Left integration boundary
        right_bound: Right integration boundary

    Returns:
        area: Integrated area under the curve (only positive values)
    """
    mask = (x_data >= left_bound) & (x_data <= right_bound)
    if np.sum(mask) == 0:
        return 0.0

    # Only integrate positive values (above baseline)
    y_positive = np.maximum(y_data[mask], 0)
    return np.trapezoid(y_positive, x_data[mask])


def calculate_molecular_weight_averages(x_mw, y_signal, peak_ranges):
    """
    Calculate molecular weight averages (Mn, Mw, Đ) for integration regions
    Only uses positive signal values (above baseline)

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

            # Ensure we only use positive values (above baseline)
            y_region = np.maximum(y_region, 0)

            # Calculate molecular weight averages
            # Number average molecular weight (Mn)
            mn = np.trapezoid(y_region, x_region) / np.trapezoid(y_region / x_region, x_region)

            # Weight average molecular weight (Mw)
            mw = np.trapezoid(y_region * x_region, x_region) / np.trapezoid(y_region, x_region)

            # Dispersity (Đ)
            dispersity = mw / mn if mn > 0 else 0

            results.append({
                'Peak': peak_name,
                'Mn (g/mol)': int(mn),
                'Mw (g/mol)': int(mw),
                'Đ': f"{dispersity:.2f}"
            })

    return pd.DataFrame(results)


def setup_integration_ui(peak_names, x_axis_type, x_plot_data, y_corrected_data,
                         peak_integration_ranges, integration_enabled):
    """
    Set up the peak integration user interface in the sidebar

    Args:
        peak_names: List of peak names from deconvolution
        x_axis_type: Type of x-axis ("MW" or "RT")
        x_plot_data: X-axis data for plotting
        y_corrected_data: Baseline-corrected y data
        peak_integration_ranges: Dictionary to store integration ranges
        integration_enabled: Whether integration is enabled

    Returns:
        total_area: Total integrated area across all peaks
        peak_integration_ranges: Updated integration ranges dictionary
    """
    total_area = 0

    if integration_enabled:
        if peak_names and x_plot_data is not None:
            st.write("Select peaks to integrate and specify integration ranges:")

            # Initialize integration ranges if not exists
            if not peak_integration_ranges:
                for name in peak_names:
                    peak_integration_ranges[name] = {"enabled": False, "left": 0, "right": 0}

            # Set appropriate step based on x-axis type
            step_val = 100.0 if x_axis_type == "MW" else 0.1

            # Create integration controls for each peak
            for i, peak_name in enumerate(peak_names):
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                with col1:
                    enabled = st.checkbox(
                        f"Integrate {peak_name}",
                        value=peak_integration_ranges[peak_name]["enabled"],
                        key=f"integrate_{peak_name}"
                    )
                    peak_integration_ranges[peak_name]["enabled"] = enabled

                if enabled:
                    # Get default values based on peak position (placeholder)
                    default_value = 1000 if x_axis_type == "MW" else 10.0

                    with col2:
                        left_bound = st.number_input(
                            f"Left bound {peak_name}",
                            value=peak_integration_ranges[peak_name].get("left", default_value * 0.8),
                            step=step_val,
                            key=f"left_{peak_name}"
                        )
                        peak_integration_ranges[peak_name]["left"] = left_bound

                    with col3:
                        right_bound = st.number_input(
                            f"Right bound {peak_name}",
                            value=peak_integration_ranges[peak_name].get("right", default_value * 1.2),
                            step=step_val,
                            key=f"right_{peak_name}"
                        )
                        peak_integration_ranges[peak_name]["right"] = right_bound

                    with col4:
                        # Calculate and display area (only positive values)
                        area = integrate_peak_region(
                            x_plot_data,
                            y_corrected_data,
                            left_bound,
                            right_bound
                        )
                        st.metric(f"Area {peak_name}", f"{area:.4f}")
                        total_area += area

            # Display total integrated area and percentage breakdown
            if total_area > 0:
                st.metric("Total Integrated Area", f"{total_area:.4f}")

                # Calculate and display percentages for each peak
                st.write("Area percentages:")
                for i, peak_name in enumerate(peak_names):
                    if peak_integration_ranges[peak_name]["enabled"]:
                        left = peak_integration_ranges[peak_name]["left"]
                        right = peak_integration_ranges[peak_name]["right"]
                        area = integrate_peak_region(x_plot_data, y_corrected_data, left, right)
                        percentage = (area / total_area) * 100 if total_area > 0 else 0
                        st.write(f"{peak_name}: {percentage:.1f}%")

        else:
            st.info("Run deconvolution first to enable peak integration")

    return total_area, peak_integration_ranges