import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import pandas as pd
import streamlit as st


def run_deconvolution(
        data_array,
        calib_array,
        mw_lim=(1e3,1e7),
        y_lim=(-0.02,1),
        n_peaks=4,
        plot_sum=False,
        manual_peaks=None,
        peaks_are_mw=True,
        peak_names=None,
        peak_width_range=(100,450),
        baseline_method="quadratic",
        baseline_ranges=None
    ):
    """
    Perform Gaussian deconvolution of a GPC trace.

    Parameters
    ----------
    data_array : ndarray
        Two-column chromatogram [RT, response].
    calib_array : ndarray
        Two-column calibration [RT, log10(MW)].
    mw_lim : tuple
        Molecular weight limits (min, max).
    y_lim : tuple
        y-axis limits for plotting.
    n_peaks : int
        Number of peaks to detect (ignored if manual_peaks provided).
    plot_sum : bool
        If True, overlay sum of fitted Gaussians.
    manual_peaks : list or None
        User-provided peaks (MW or RT, depending on peaks_are_mw).
    peaks_are_mw : bool
        If True, manual_peaks are MW; else retention time.
    peak_names : list or None
        Names for peaks.
    peak_width_range : tuple
        Range (lo,hi) of fitting window widths to try.
    baseline_method : str
        'flat', 'linear', 'quadratic'.
    baseline_ranges : list
        List of [MW_min, MW_max] ranges for baseline calculation.

    Returns
    -------
    fig : matplotlib Figure
    results_df : pandas DataFrame
        Table of peak Mn and area %.
    """

    # --- 1. calibration
    rt_cal, logmw_cal = calib_array[:,0], calib_array[:,1]
    f_logmw = interp1d(rt_cal, logmw_cal, kind="linear", fill_value="extrapolate")
    f_rt    = interp1d(logmw_cal, rt_cal, kind="linear", fill_value="extrapolate")
    mw2rt   = lambda mw: f_rt(np.log10(mw))

    # --- 2. retention time window
    rt_min, rt_max = mw2rt(mw_lim[1]), mw2rt(mw_lim[0])  # hi MW → low RT
    mask_window = (data_array[:,0]>=rt_min) & (data_array[:,0]<=rt_max)

    # --- 3. extract + normalize
    x_rt = data_array[:,0].astype(float)
    y_raw = data_array[:,1].astype(float)
    y_norm = y_raw / y_raw[mask_window].max()

    # --- 4. baseline correction
    def baseline_correction(x_rt_in,y_in,x_mw_in):
        n_required = dict(flat=1,linear=2,quadratic=3)[baseline_method]
        if baseline_ranges is None or len(baseline_ranges)!=n_required:
            raise ValueError(f"{baseline_method} requires {n_required} baseline ranges")
        refs=[]
        for rng in baseline_ranges:
            m=(x_mw_in>=rng[0])&(x_mw_in<=rng[1])
            if not np.any(m):
                raise ValueError(f"No points in baseline MW range {rng}")
            refs.append((x_rt_in[m].mean(),y_in[m].mean()))
        xs,ys=zip(*refs)
        deg=dict(flat=0,linear=1,quadratic=2)[baseline_method]
        coeffs=np.polyfit(xs,ys,deg)
        baseline=np.polyval(coeffs,x_rt_in)
        return y_in-baseline, baseline

    x_mw=10**f_logmw(x_rt)
    y_corr, baseline=baseline_correction(x_rt,y_norm,x_mw)

    # --- 5. peak detection
    if manual_peaks:
        x_peaks_rt=[]
        for pk in manual_peaks:
            rt_est=mw2rt(pk) if peaks_are_mw else pk
            x_peaks_rt.append(rt_est)
        n_peaks=len(x_peaks_rt)
    else:
        idx,_=find_peaks(y_corr,distance=200,width=50)
        if len(idx)<n_peaks:
            n_peaks=len(idx)
        idx=idx[np.argsort(y_corr[idx])[-n_peaks:]]
        x_peaks_rt=x_rt[idx]

    # --- 6. Gaussian fitting
    def gaussian(x,a,mu,s): return a*np.exp(-(x-mu)**2/(2*s**2))

    best_resid=np.inf
    best_gaussians,best_params=None,None
    for width in range(peak_width_range[0],peak_width_range[1]):
        y_left=y_corr.copy()
        gaussians,params=[],[]
        try:
            for mu_guess in x_peaks_rt:
                i0=np.argmin(np.abs(x_rt-mu_guess))
                lo,hi=max(0,i0-width),min(len(x_rt),i0+width)
                p0=[y_left[i0], mu_guess, 0.1]
                popt,_=curve_fit(gaussian,x_rt[lo:hi],y_left[lo:hi],p0=p0)
                g=gaussian(x_rt,*popt)
                gaussians.append(g); params.append(popt)
                y_left-=g
            resid=np.abs(y_left).sum()
            if resid<best_resid:
                best_resid,resid; best_gaussians, best_params = resid,params,gaussians
        except Exception: continue

    if not best_params:
        raise RuntimeError("Peak fitting failed")

    mus=[p[1] for p in best_params]
    mw_vals=10**f_logmw(mus)
    areas=[trapezoid(g,x_rt) for g in best_gaussians]
    pct=np.array(areas)/sum(areas)*100

    order=np.argsort(mw_vals)[::-1]
    mw_vals,pct=np.array(mw_vals)[order],pct[order]
    best_gaussians=np.array(best_gaussians)[order]

    if peak_names is None: peak_names=[]
    peak_names=(peak_names+[f"Peak {i+1}" for i in range(len(mw_vals))])[:len(mw_vals)]

    # --- 7. plot
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(x_mw,y_corr,color="#ef476f",lw=2,label="Trace (baseline corr.)")
    colors=['#FFbf00','#06d6a0','#118ab2','#073b4c']
    for i,(g,pc) in enumerate(zip(best_gaussians,pct)):
        ax.plot(x_mw,g,color=colors[i%len(colors)],label=f"{peak_names[i]}: {pc:.1f}%")
    if plot_sum:
        ax.plot(x_mw,best_gaussians.sum(axis=0),'k--',lw=1.2,label="Sum")
    ax.set(xscale="log",xlim=mw_lim,ylim=y_lim,
           xlabel="Molecular weight (g/mol)",ylabel="Normalized Response")
    ax.legend(); ax.grid(False); fig.tight_layout()

    results=pd.DataFrame({"Peak":peak_names,"Mn (g/mol)":mw_vals.astype(int),"Area %":pct.round(1)})
    return fig, results
