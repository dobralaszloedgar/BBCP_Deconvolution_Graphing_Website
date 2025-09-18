# gaussian_deconvolution.py
from __future__ import annotations

import io
import time
import hashlib
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional autorefresh dependency with safe fallback
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    _AUTOREFRESH_AVAILABLE = True
except Exception:
    _AUTOREFRESH_AVAILABLE = False

# Import the deconvolution engine and flag from the shared module
from Deconvolution import run_deconvolution, PYBASELINES_AVAILABLE  # noqa: E402


# ----------------------------- Utilities ----------------------------- #

def _read_table_file(uploaded) -> Optional[np.ndarray]:
    """
    Read a two-column numeric file (CSV/TSV/whitespace).
    Returns np.ndarray of shape (N, 2) or None if not available/parsable.
    """
    if uploaded is None:
        return None
    content = uploaded.read()
    uploaded.seek(0)
    buf = io.BytesIO(content)
    # Try several separators
    for sep in (None, r"\s+", ",", "\t"):
        try:
            buf.seek(0)
            df = pd.read_csv(buf, sep=sep, engine="python", header=None, comment="#")
            if df.shape[1] >= 2:
                arr = df.iloc[:, :2].to_numpy(dtype=float)
                return arr
        except Exception:
            continue
    return None


def _parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    vals: List[float] = []
    for chunk in text.replace(";", ",").split(","):
        s = chunk.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            pass
    return vals


def _config_signature(**kwargs) -> str:
    """
    Build a deterministic, order-stable signature of the current config to detect changes.
    Handles nested dicts/lists/tuples/sets, bytes, and NumPy-like arrays without
    hashing entire large buffers (uses shape/dtype and head/tail sampling).
    """
    import hashlib

    def _to_bytes(obj) -> bytes:
        # Primitives and direct bytes
        if obj is None:
            return b'null'
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj)
        if isinstance(obj, (str, int, bool)):
            return repr(obj).encode("utf-8")
        if isinstance(obj, float):
            # Normalize -0.0 and ensure stable textual form
            if obj == 0.0:
                obj = 0.0
            return repr(float(obj)).encode("utf-8")

        # NumPy-like arrays (duck-typed)
        if hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "tobytes"):
            try:
                h = hashlib.sha256()
                h.update(repr(getattr(obj, "shape", None)).encode("utf-8"))
                h.update(repr(getattr(obj, "dtype", None)).encode("utf-8"))
                # Try zero-copy bytes view
                try:
                    mv = memoryview(obj).cast("B")
                    total = len(mv)
                    if total <= 256:
                        h.update(mv.tobytes())
                    else:
                        h.update(mv[:128].tobytes())
                        h.update(mv[-128:].tobytes())
                except Exception:
                    bb = obj.tobytes()
                    if len(bb) <= 256:
                        h.update(bb)
                    else:
                        h.update(bb[:128])
                        h.update(bb[-128:])
                return h.digest()
            except Exception:
                return repr(obj).encode("utf-8")

        # Mappings (sorted by key repr for determinism)
        if isinstance(obj, dict):
            h = hashlib.sha256()
            h.update(b"{")
            for i, key in enumerate(sorted(obj.keys(), key=lambda x: repr(x))):
                if i:
                    h.update(b",")
                h.update(_to_bytes(key))
                h.update(b":")
                h.update(_to_bytes(obj[key]))
            h.update(b"}")
            return h.digest()

        # Sets/frozensets (order-independent via sorting of item digests)
        if isinstance(obj, (set, frozenset)):
            item_bytes = sorted((_to_bytes(x) for x in obj))
            h = hashlib.sha256()
            h.update(b"(")
            for i, bts in enumerate(item_bytes):
                if i:
                    h.update(b",")
                h.update(bts)
            h.update(b")")
            return h.digest()

        # Sequences
        if isinstance(obj, (list, tuple)):
            h = hashlib.sha256()
            h.update(b"[" if isinstance(obj, list) else b"(")
            for i, item in enumerate(obj):
                if i:
                    h.update(b",")
                h.update(_to_bytes(item))
            h.update(b"]" if isinstance(obj, list) else b")")
            return h.digest()

        # Fallback
        return repr(obj).encode("utf-8")

    m = hashlib.sha256()
    for k in sorted(kwargs.keys(), key=lambda x: repr(x)):
        m.update(_to_bytes(k))
        m.update(b"=")
        m.update(_to_bytes(kwargs[k]))
        m.update(b";")
    return m.hexdigest()



def _auto_refresh_every(ms: int, key: str = "autorefresh_tick"):
    """
    Trigger reruns every ms milliseconds.
    Uses streamlit-autorefresh if available, else a minimal JS fallback.
    """
    if _AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=ms, debounce=True, key=key)
    else:
        # Fallback: simple JS reload (no debounce)
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {ms});</script>",
            unsafe_allow_html=True,
        )


# ------------------------------- App --------------------------------- #

def main():
    # Do NOT call st.set_page_config here; the launcher already sets it.
    st.title("Gaussian Deconvolution")

    # Periodic rerun every 3 seconds, with debounce when available
    _auto_refresh_every(3000, key="gauss_autorefresh")

    # Session state for throttling and change detection
    if "lastupdatetime" not in st.session_state:
        st.session_state.lastupdatetime = 0.0
    if "config_sig" not in st.session_state:
        st.session_state.config_sig = ""

    # --------------------------- Sidebar --------------------------- #
    with st.sidebar:
        st.header("Data")
        data_file = st.file_uploader("Upload data (two columns: x, y)", type=["csv", "tsv", "txt"])
        calib_file = st.file_uploader("Upload calibration (RT, log10(MW))", type=["csv", "tsv", "txt"])

        st.header("X-axis")
        x_axis_type = st.radio("X axis type", options=["MW", "RT"], index=0, horizontal=True)

        if x_axis_type == "MW":
            x_min = st.number_input("MW min (g/mol)", value=1e3, format="%.6g")
            x_max = st.number_input("MW max (g/mol)", value=1e7, format="%.6g")
            x_lim = [float(x_min), float(x_max)]
            x_label_preview = "MW"
        else:
            x_min = st.number_input("RT min (min)", value=0.0)
            x_max = st.number_input("RT max (min)", value=30.0)
            x_lim = [float(x_min), float(x_max)]
            x_label_preview = "RT"

        st.header("Y-axis")
        y_min = st.number_input("Y min", value=-0.02)
        y_max = st.number_input("Y max", value=1.0)
        y_lim = [float(y_min), float(y_max)]

        st.header("Peaks")
        n_peaks = int(st.number_input("Number of peaks", min_value=1, max_value=12, value=4, step=1))
        plot_sum = st.checkbox("Plot sum of Gaussians", value=False)

        manual_peaks_raw = st.text_input(f"Manual peaks ({x_label_preview}, comma/semicolon separated)", value="")
        manual_peaks = _parse_float_list(manual_peaks_raw)
        peaks_are_mw = st.checkbox("Manual peaks are MW values (if MW mode)", value=True)

        # Names and colors
        default_names = [f"Peak {i+1}" for i in range(n_peaks)]
        peak_names_text = st.text_input("Peak names (comma separated)", value=", ".join(default_names))
        peak_names = [s.strip() for s in peak_names_text.split(",") if s.strip()]
        if len(peak_names) < n_peaks:
            peak_names += [f"Peak {i+1}" for i in range(len(peak_names), n_peaks)]
        if len(peak_names) > n_peaks:
            peak_names = peak_names[:n_peaks]

        st.markdown("Colors (hex)")
        default_colors = ['#FFbf00', '#06d6a0', '#118ab2', '#073b4c',
                          '#a83232', '#a832a8', '#32a852', '#3264a8',
                          '#a86432', '#6432a8', '#2ca02c', '#d62728']
        peak_colors: List[str] = []
        for i in range(n_peaks):
            col = st.text_input(f"Color for {peak_names[i]}", value=default_colors[i % len(default_colors)], key=f"col_{i}")
            peak_colors.append(col)

        st.header("Peak width search")
        c1, c2 = st.columns(2)
        with c1:
            wmin = int(st.number_input("Width min (index)", value=100, min_value=10, step=10))
        with c2:
            wmax = int(st.number_input("Width max (index)", value=400, min_value=wmin+1, step=10))
        peak_width_range = [wmin, wmax]

        st.header("Baseline")
        baseline_options = ["None", "flat", "linear", "quadratic"]
        if PYBASELINES_AVAILABLE:
            baseline_options.append("arpls")
        baseline_method = st.selectbox("Method", options=baseline_options, index=0)

        required_ranges = {"None": 0, "flat": 1, "linear": 2, "quadratic": 3, "arpls": 0}[baseline_method]
        baseline_ranges: List[Tuple[float, float]] = []
        for i in range(required_ranges):
            c1, c2 = st.columns(2)
            with c1:
                xmin = st.number_input(f"Baseline range {i+1} min ({x_label_preview})", value=0.0, key=f"bl_min_{i}")
            with c2:
                xmax = st.number_input(f"Baseline range {i+1} max ({x_label_preview})", value=1.0, key=f"bl_max_{i}")
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            baseline_ranges.append((float(xmin), float(xmax)))

        st.header("Styles")
        original_data_label = st.text_input("Original data label", value="Original Data")
        original_data_color = st.text_input("Original data color (hex)", value="#ef476f")
        font_family = st.text_input("Font family", value="Times New Roman")
        font_size = int(st.number_input("Font size", value=12, min_value=6, max_value=48))
        fig_w = float(st.number_input("Figure width (in)", value=8.0, step=0.5))
        fig_h = float(st.number_input("Figure height (in)", value=5.0, step=0.5))
        fig_size = (fig_w, fig_h)

        x_label_text = st.text_input("X label (leave blank for default)", value="")
        y_label_text = st.text_input("Y label", value="Normalized Response")
        x_label_style = st.selectbox("X label style", ["normal", "italic", "bold", "italic+bold"], index=0)
        y_label_style = st.selectbox("Y label style", ["normal", "italic", "bold", "italic+bold"], index=0)
        legend_style = st.selectbox("Legend style", ["normal", "italic", "bold", "italic+bold"], index=0)

    # --------------------------- Main area --------------------------- #
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Inputs")
        data_arr = _read_table_file(data_file)
        if data_arr is not None:
            st.write(f"Data points: {data_arr.shape[0]} rows")
        else:
            st.info("Upload a two-column data file to begin.")
        calib_arr = _read_table_file(calib_file)

        if x_axis_type == "MW" and calib_arr is None:
            st.warning("MW mode selected: provide calibration (RT, log10(MW)) to compute MW axis.")

    with right:
        st.subheader("Status")

        # Build a signature of inputs; triggers ASAP render on change
        sig = _config_signature(
            data_hash=("none" if data_arr is None else str(data_arr.shape) + str(data_arr[:2].tolist())),
            calib_hash=("none" if calib_arr is None else str(calib_arr.shape) + str(calib_arr[:2].tolist())),
            x_axis_type=x_axis_type,
            x_lim=x_lim,
            y_lim=y_lim,
            n_peaks=n_peaks,
            plot_sum=plot_sum,
            manual_peaks=tuple(manual_peaks),
            peaks_are_mw=peaks_are_mw,
            peak_names=tuple(peak_names),
            peak_colors=tuple(peak_colors),
            peak_width_range=tuple(peak_width_range),
            baseline_method=baseline_method,
            baseline_ranges=tuple(tuple(r) for r in baseline_ranges),
            original_data_label=original_data_label,
            original_data_color=original_data_color,
            font_family=font_family,
            font_size=font_size,
            fig_size=fig_size,
            x_label_text=x_label_text,
            y_label_text=y_label_text,
            x_label_style=x_label_style,
            y_label_style=y_label_style,
            legend_style=legend_style,
        )

        # If configuration changed, force an ASAP render (reset throttle)
        if sig != st.session_state.config_sig:
            st.session_state.config_sig = sig
            st.session_state.lastupdatetime = 0.0

        now = time.time()
        last = float(st.session_state.lastupdatetime or 0.0)
        should_update = (last == 0.0) or ((now - last) >= 3.0)

        if should_update and data_arr is not None and (x_axis_type == "RT" or calib_arr is not None):
            try:
                fig, results_df = run_deconvolution(
                    data_array=data_arr,
                    calib_array=calib_arr,
                    x_axis_type=x_axis_type,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    n_peaks=n_peaks,
                    plot_sum=plot_sum,
                    manual_peaks=manual_peaks,
                    peaks_are_mw=peaks_are_mw,
                    peak_names=peak_names,
                    peak_colors=peak_colors,
                    peak_width_range=[int(peak_width_range[0]), int(peak_width_range[1])],
                    baseline_method=baseline_method,
                    baseline_ranges=baseline_ranges,
                    original_data_color=original_data_color,
                    original_data_label=original_data_label,
                    font_family=font_family,
                    font_size=font_size,
                    fig_size=fig_size,
                    x_label=x_label_text,
                    y_label=y_label_text,
                    x_label_style=x_label_style,
                    y_label_style=y_label_style,
                    legend_style=legend_style,
                )

                st.session_state.lastupdatetime = now

                st.subheader("Deconvolution Plot")
                st.pyplot(fig, clear_figure=True)

                st.subheader("Results")
                st.dataframe(results_df, use_container_width=True)

                if results_df is not None and len(results_df) > 0:
                    csv_bytes = results_df.to_csv(index=False).encode()
                    st.download_button(
                        "Download results CSV",
                        data=csv_bytes,
                        file_name=f"deconvolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Deconvolution error: {e}")

        else:
            if data_arr is None:
                st.info("Waiting for data upload...")
            elif x_axis_type == "MW" and calib_arr is None:
                st.info("Waiting for calibration for MW mode...")
            else:
                remaining = max(0.0, 3.0 - (now - last)) if last > 0.0 else 0.0
                st.caption(f"Next auto-update in {remaining:.1f} s")

    st.caption("Auto-updates every 3 seconds; heavy redraw is throttled to timer ticks to avoid re-rendering after each change.")


if __name__ == "__main__":
    main()
