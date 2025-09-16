import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

# ----------  core algorithm wrapped in a single function -------------
def run_deconvolution(
        data_array,
        calib_array,
        mw_lim,
        y_lim,
        n_peaks,
        plot_sum,
        manual_peaks,
        peaks_are_mw,
        peak_names,
        peak_width_range,
        baseline_method,
        baseline_ranges
    ):
    """Return matplotlib Figure and deconvolution results table."""

    # 1. Build MW↔RT interpolators from calibration
    rt_cal, logmw_cal = calib_array[:,0], calib_array[:,1]
    f_logmw = interp1d(rt_cal, logmw_cal, fill_value='extrapolate')
    f_rt    = interp1d(logmw_cal, rt_cal, fill_value='extrapolate')
    mw2rt   = lambda mw: f_rt(np.log10(mw))

    # 2. Establish RT window from desired MW limits
    rt_min, rt_max = mw2rt(mw_lim[1]), mw2rt(mw_lim[0])      # remember: high-MW elutes early
    rt_window = (rt_min, rt_max)

    # 3. Pull first chromatogram (col0=RT, col1=response)
    x_rt  = data_array[:,0].astype(float)
    y_raw = data_array[:,1].astype(float)

    # 4. Normalize within RT window
    mask_window = (x_rt>=rt_window[0]) & (x_rt<=rt_window[1])
    y_norm = y_raw / y_raw[mask_window].max()

    # 5. Baseline correction helper ------------------------------------
    def baseline_correction(x_rt_in, y_in, x_mw_in):
        refs = []
        required = dict(flat=1, linear=2, quadratic=3)[baseline_method]
        if len(baseline_ranges)!=required:
            st.error(f"{baseline_method} needs {required} MW ranges"); st.stop()
        for rng in baseline_ranges:
            m = (x_mw_in>=rng[0]) & (x_mw_in<=rng[1])
            refs.append((x_rt_in[m].mean(), y_in[m].mean()))
        xs, ys = zip(*refs)
        degree = dict(flat=0, linear=1, quadratic=2)[baseline_method]
        coeffs = np.polyfit(xs, ys, degree)
        baseline = np.polyval(coeffs, x_rt_in)
        return y_in-baseline, baseline
    # ------------------------------------------------------------------

    # Convert x-axis to MW for later plotting
    x_mw = 10**f_logmw(x_rt)

    # Apply baseline correction
    y_corr, baseline = baseline_correction(x_rt, y_norm, x_mw)

    # 6. Determine peaks (manual or automatic)
    if manual_peaks:
        x_peaks_rt = []
        for p in manual_peaks:
            rt_est = mw2rt(p) if peaks_are_mw else p
            x_peaks_rt.append(rt_est)
        n_peaks = len(x_peaks_rt)
    else:
        idx,_ = find_peaks(y_corr, distance=200, width=50)
        idx = idx[np.argsort(y_corr[idx])[-n_peaks:]]     # top n by height
        x_peaks_rt = x_rt[idx]

    # 7. Fit multiple Gaussians ----------------------------------------
    def gaussian(x,a,mu,s):
        return a*np.exp(-(x-mu)**2/(2*s**2))

    best_resid = np.inf
    best_gaussians, best_params = None, None
    for width in range(*peak_width_range):
        y_left = y_corr.copy()
        gaussians, params = [], []
        try:
            for mu_guess in x_peaks_rt:
                idx_center = np.argmin(np.abs(x_rt-mu_guess))
                lo, hi = max(0,idx_center-width), min(len(x_rt),idx_center+width)
                p0=[y_left[idx_center], mu_guess, 0.1]
                popt,_ = curve_fit(gaussian, x_rt[lo:hi], y_left[lo:hi], p0=p0)
                g = gaussian(x_rt,*popt)
                gaussians.append(g); params.append(popt)
                y_left -= g
            resid = np.abs(y_left).sum()
            if resid < best_resid:
                best_resid, best_gaussians, best_params = resid, gaussians, params
        except Exception: continue
    # ------------------------------------------------------------------

    if best_params is None:
        st.error("Peak fitting failed."); st.stop()

    mus = [p[1] for p in best_params]
    mw_vals = 10**f_logmw(mus)

    # Areas (%)
    areas = [trapezoid(g,x_rt) for g in best_gaussians]
    pct   = np.array(areas)/sum(areas)*100

    # Sort descending MW
    order = np.argsort(mw_vals)[::-1]
    mw_vals, pct, best_gaussians = np.array(mw_vals)[order], pct[order], np.array(best_gaussians)[order]
    peak_names = (peak_names + [f"Peak {i+1}" for i in range(len(mw_vals))])[:len(mw_vals)]

    # 8. Build figure
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(x_mw, y_corr, color="#ef476f", lw=2, label="Baseline-corr. trace")
    colors=['#FFbf00','#06d6a0','#118ab2','#073b4c']
    for i,(g,pc) in enumerate(zip(best_gaussians,pct)):
        ax.plot(x_mw, g, color=colors[i%len(colors)],
                label=f"{peak_names[i]}: {pc:.1f}%")
    if plot_sum:
        ax.plot(x_mw, best_gaussians.sum(axis=0),'k--',lw=1.2,label="Sum of Gaussians")
    ax.set(xscale="log", xlim=mw_lim, ylim=y_lim,
           xlabel="Molecular weight (g/mol)",
           ylabel="Normalized response")
    ax.legend(); ax.grid(False); fig.tight_layout()

    # 9. Prepare results table
    import pandas as pd
    results_df = pd.DataFrame({
        "Peak":peak_names,
        "Mn (g/mol)":mw_vals.astype(int),
        "Area %":pct.round(1)
    })
    return fig, results_df