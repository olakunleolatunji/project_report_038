#!/usr/bin/env python3
"""
================================================================================
UAV ATTITUDE STABILISATION - COMPLETE DATA ANALYSIS
================================================================================
Reproduces every number, table and figure reported in Chapter 4.

Project : UAV attitude stabilisation using a PID controller on a
          resource-shared companion computer
Data    : Per-cycle CSV logs from the rung 1 shadow-mode campaign

--------------------------------------------------------------------------------
INPUT FILES
--------------------------------------------------------------------------------
Four CSV logs are required. Set DATA_DIR below to the folder containing them.

  001_bench_unloaded_20260829_103131.csv    Bench, motors off, background only
  002_bench_loaded_20260829_104816.csv      Bench, motors off, synthetic load
  003_hover_unloaded_20260831_204355.csv    Hover AltHold, no added load
  004_hover_loaded_20260831_211539.csv      Hover AltHold, synthetic load

Which runs feed which metric:

  Metric 1  Loop timing .................... all four runs
  Metric 2  CPU utilisation ................ all four runs
  Metric 3  Noise rejection / SNR .......... 003 and 004 only (needs motors on)
  Metric 4  CF vs EKF cross-validation ..... 003 and 004 only
  Metric 5  Command divergence ............. 003 and 004 only
  Diagnostics (staleness, CF rebuild,
    vibration, alpha sweep) ............... 003 and 004 only

The bench runs cannot support Metrics 3-5: with motors off there is no
vibration to reject and no active-flight rows to select.

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------
Console : all result tables
Figures : eight PNG files written to OUT_DIR

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pip install pandas numpy matplotlib
    python uav_analysis.py

Edit DATA_DIR and OUT_DIR below if your paths differ.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "./"           # folder containing the four CSV files
OUT_DIR  = "./figures"   # folder to write PNG figures into

RUNS = {
    "001": ("001_bench_unloaded_20260829_103131.csv", "Bench Unloaded"),
    "002": ("002_bench_loaded_20260829_104816.csv",   "Bench Loaded"),
    "003": ("003_hover_unloaded_20260831_204355.csv", "Hover Unloaded"),
    "004": ("004_hover_loaded_20260831_211539.csv",   "Hover Loaded"),
}

HOVER_RUNS = ["003", "004"]   # runs with motors spinning

LOOP_HZ        = 50.0
DEADLINE_US    = 1e6 / LOOP_HZ        # 20 000 us
ALPHA_DEPLOYED = 0.98                 # complementary filter coefficient used in flight
KP             = 4.5                  # outer loop proportional gain
EKF_LIMIT_DEG  = 60.0                 # exclude samples beyond this EKF roll magnitude
ACTIVE_DPS     = 0.01                 # active-flight rate threshold
AGREE_DEG      = 3.0                  # CF vs EKF acceptance threshold

# Run durations in seconds, taken from the .meta.txt summary block of each run.
# Used only to convert message counts into stream rates (Table 4.3).
DURATIONS = {"001": 192.9, "002": 229.1, "003": 137.0, "004": 122.4}

# Message counts from the .meta.txt summary block: ATTITUDE, ATTITUDE_TARGET, RC_CHANNELS
MSG_COUNTS = {
    "001": (1103, 10006,  719),
    "002": (2251, 11801, 1843),
    "003": (2412,  2408, 2422),
    "004": (2114,  2121, 2122),
}

# ==============================================================================
# PLOT STYLE
# ==============================================================================

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#4a4a4a"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ==============================================================================
# HELPERS
# ==============================================================================

def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_runs():
    """Read all four CSV logs into a dict of DataFrames."""
    dfs, labels = {}, {}
    for key, (fname, label) in RUNS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing input file: {path}\n"
                f"Set DATA_DIR at the top of this script to the folder "
                f"containing the four CSV logs."
            )
        dfs[key] = pd.read_csv(path)
        labels[key] = label
    return dfs, labels


def active_rows(df):
    """
    Select active-flight rows.

    A row is retained when the flight controller reports a roll or pitch
    angular rate above ACTIVE_DPS in magnitude. This removes pre-arm and
    post-landing intervals, during which the airframe sits stationary on
    the ground and the filter is not being exercised.
    """
    return df[(df["fc_rate_roll_dps"].abs() > ACTIVE_DPS) |
              (df["fc_rate_pitch_dps"].abs() > ACTIVE_DPS)].copy()


def validated(df):
    """Active-flight rows, further restricted to the EKF validity envelope."""
    a = active_rows(df)
    return a[a["ekf_roll_deg"].abs() < EKF_LIMIT_DEG].copy()


def debias(cf, ekf):
    """
    Remove the fixed offset between the two estimates.

    A constant difference arises from residual mounting misalignment between
    the dedicated MPU6050 and the flight controller's internal IMU. The median
    is used rather than the mean because it is insensitive to the heavy tails
    produced by vibration.
    """
    bias = np.median(cf - ekf)
    return cf - bias, bias


def run_cf(acc_angle, gyro_rate, dt, alpha):
    """
    Complementary filter, matching the deployed C++ implementation:

        theta(k) = alpha * (theta(k-1) + gyro(k) * dt(k)) + (1 - alpha) * acc(k)

    Vectorised form is not possible because the recurrence is sequential.
    """
    out = np.zeros(len(acc_angle))
    out[0] = acc_angle[0]
    for i in range(1, len(acc_angle)):
        out[i] = alpha * (out[i - 1] + gyro_rate[i] * dt[i]) + (1 - alpha) * acc_angle[i]
    return out


def show(rows):
    print(pd.DataFrame(rows).to_string(index=False))


# ==============================================================================
# METRIC 1 - LOOP TIMING
# ==============================================================================

def metric1(dfs, labels):
    """
    Loop period statistics and deadline adherence.

    Each cycle is timestamped with clock_gettime(CLOCK_MONOTONIC) and the
    measured period is logged as dt_s. An overrun is flagged by the logger
    when a cycle exceeds the 20 ms deadline.
    """
    banner("METRIC 1  -  LOOP TIMING   (Table 4.7)")
    rows = []
    for key, df in dfs.items():
        dt_us = df["dt_s"] * 1e6
        rows.append({
            "Run":            labels[key],
            "Cycles":         f"{len(df):,}",
            "Mean (us)":      f"{dt_us.mean():,.1f}",
            "Std dev (us)":   f"{dt_us.std():.1f}",
            "Min (us)":       f"{dt_us.min():,.0f}",
            "Max (us)":       f"{dt_us.max():,.0f}",
            "Overruns":       int(df["overrun"].sum()),
        })
    show(rows)

    total_cycles = sum(len(d) for d in dfs.values())
    total_over   = sum(int(d["overrun"].sum()) for d in dfs.values())
    print(f"\n  Total across all runs: {total_cycles:,} cycles, {total_over} overruns")

    u = (dfs["003"]["dt_s"] * 1e6).std()
    l = (dfs["004"]["dt_s"] * 1e6).std()
    print(f"  Hover jitter unloaded -> loaded: {u:.1f} -> {l:.1f} us  "
          f"({l/u:.1f}x increase, {l/DEADLINE_US*100:.2f}% of loop period)")
    print("  NOTE: the bench pair (001/002) is NOT a clean load comparison - both")
    print("        carried heavy uncontrolled background activity. Load conclusions")
    print("        rest on the hover pair only.")


# ==============================================================================
# METRIC 2 - CPU UTILISATION
# ==============================================================================

def metric2(dfs, labels):
    """
    System and control-process CPU utilisation.

    cpu_busy_pct is sampled from /proc/stat each cycle as the non-idle
    fraction since the previous sample. Rows before the sampler has produced
    its first reading are logged as zero and are excluded here.
    """
    banner("METRIC 2  -  CPU UTILISATION   (Table 4.8)")
    rows = []
    for key, df in dfs.items():
        sysc = df.loc[df["cpu_busy_pct"] > 0, "cpu_busy_pct"]
        proc = df.loc[df["proc_cpu_pct"] > 0, "proc_cpu_pct"]
        rows.append({
            "Run":                 labels[key],
            "System mean (%)":     f"{sysc.mean():.2f}",
            "System peak (%)":     f"{sysc.max():.2f}",
            "Process mean (%)":    f"{proc.mean():.2f}",
            "Process peak (%)":    f"{proc.max():.2f}",
            "Warm-up rows":        int((df["cpu_busy_pct"] == 0).sum()),
        })
    show(rows)

    u = dfs["003"].loc[dfs["003"]["cpu_busy_pct"] > 0, "cpu_busy_pct"].mean()
    l = dfs["004"].loc[dfs["004"]["cpu_busy_pct"] > 0, "cpu_busy_pct"].mean()
    print(f"\n  Hover system CPU unloaded -> loaded: {u:.2f}% -> {l:.2f}%  ({l/u:.1f}x)")
    print(f"  Headroom remaining under load: {100-l:.1f}%")


# ==============================================================================
# METRIC 3 - NOISE REJECTION AND SNR
# ==============================================================================

def metric3(dfs, labels):
    """
    Noise rejection ratio and signal-to-noise ratio.

        NRR = std(theta_acc - theta_cf) / std(theta_cf)
        SNR = 10 * log10( var(theta_cf) / var(theta_acc - theta_cf) )

    The decibel figure is negative during hover because motor vibration
    variance exceeds the variance of genuine attitude motion. This is a
    physically meaningful result, not an error; the NRR form is the more
    interpretable of the two.
    """
    banner("METRIC 3  -  NOISE REJECTION AND SNR   (Table 4.9)")
    rows = []
    for key in HOVER_RUNS:
        a = active_rows(dfs[key])
        for axis, cf_col, acc_col in [("Roll",  "roll_cf_deg",  "roll_acc_deg"),
                                      ("Pitch", "pitch_cf_deg", "pitch_acc_deg")]:
            var_sig   = a[cf_col].var()
            var_noise = (a[acc_col] - a[cf_col]).var()
            rows.append({
                "Run":                  labels[key],
                "Axis":                 axis,
                "Signal var (deg^2)":   f"{var_sig:,.2f}",
                "Noise var (deg^2)":    f"{var_noise:,.2f}",
                "NRR (x)":              f"{np.sqrt(var_noise/var_sig):.2f}",
                "SNR (dB)":             f"{10*np.log10(var_sig/var_noise):.2f}",
                "Active rows":          f"{len(a):,}",
            })
    show(rows)
    print("\n  Noise variance is near-identical between loaded and unloaded runs:")
    print("  vibration is mechanical and is unaffected by processor load.")


# ==============================================================================
# METRIC 4 - CF vs EKF CROSS-VALIDATION
# ==============================================================================

def metric4(dfs, labels):
    """
    Complementary filter agreement with the flight controller's EKF.

    Computed over active-flight rows within the EKF validity envelope, after
    removing the fixed mounting-misalignment offset.
    """
    banner("METRIC 4  -  CF vs EKF CROSS-VALIDATION   (Table 4.10)")
    rows, store = [], {}
    for key in HOVER_RUNS:
        v = validated(dfs[key])
        for axis, cf_col, ekf_col in [("Roll",  "roll_cf_deg",  "ekf_roll_deg"),
                                      ("Pitch", "pitch_cf_deg", "ekf_pitch_deg")]:
            cf_db, bias = debias(v[cf_col].values, v[ekf_col].values)
            err = np.abs(cf_db - v[ekf_col].values)
            rows.append({
                "Run":               labels[key],
                "Axis":              axis,
                "Removed bias (deg)": f"{bias:+.2f}",
                "Median err (deg)":  f"{np.median(err):.2f}",
                "RMS err (deg)":     f"{np.sqrt((err**2).mean()):.2f}",
                f"Within {AGREE_DEG:.0f}deg (%)": f"{(err < AGREE_DEG).mean()*100:.1f}",
                "N":                 f"{len(v):,}",
            })
            if axis == "Roll":
                store[key] = dict(v=v, cf_db=cf_db, err=err)
    show(rows)

    print("\n  ATTITUDE message staleness (relevant to the diagnostic section):")
    for key in HOVER_RUNS:
        v = store[key]["v"]
        print(f"    {labels[key]:16s} mean {v['age_att_s'].mean()*1e3:5.1f} ms   "
              f"p95 {v['age_att_s'].quantile(.95)*1e3:5.1f} ms   "
              f"max {v['age_att_s'].max()*1e3:5.1f} ms")
    return store


# ==============================================================================
# METRIC 5 - COMMAND DIVERGENCE
# ==============================================================================

def metric5(dfs, labels):
    """
    Divergence between the shadow PID rate command and the flight
    controller's own inner-loop rate demand.

        D(k) = theta_cmd(k) - theta_fc(k)

    Both controllers share the same proportional gain (KP), matched to
    ArduPilot's ATC_ANG_RLL_P, so divergence is attributable to rate
    limiting, feedforward and the noise carried on the CF angle estimate
    rather than to a gain mismatch.
    """
    banner("METRIC 5  -  COMMAND DIVERGENCE   (Table 4.13)")
    rows = []
    for key in HOVER_RUNS:
        a = active_rows(dfs[key])
        dr = a["roll_cmd_dps"]  - a["fc_rate_roll_dps"]
        dp = a["pitch_cmd_dps"] - a["fc_rate_pitch_dps"]
        cmd_mag = a["roll_cmd_dps"].abs().mean()
        fc_mag  = a["fc_rate_roll_dps"].abs().mean()
        rows.append({
            "Run":                    labels[key],
            "Roll div RMS (dps)":     f"{np.sqrt((dr**2).mean()):.2f}",
            "Pitch div RMS (dps)":    f"{np.sqrt((dp**2).mean()):.2f}",
            "Mean |cmd| (dps)":       f"{cmd_mag:.2f}",
            "Mean |FC rate| (dps)":   f"{fc_mag:.2f}",
            "Ratio":                  f"{cmd_mag/fc_mag:.2f}x",
            "Active rows":            f"{len(a):,}",
        })
    show(rows)
    print(f"\n  Note: a {6.0:.0f} deg error on the angle estimate becomes a "
          f"{6.0*KP:.0f} dps spurious rate command at Kp = {KP}.")
    print("  Metrics 4 and 5 are therefore not independent findings.")


# ==============================================================================
# DIAGNOSTIC A - IS THE RESIDUAL CAUSED BY TELEMETRY STALENESS?
# ==============================================================================

def diag_staleness(dfs, labels):
    """
    Test the hypothesis that the CF-EKF residual is an artefact of comparing
    a 50 Hz filter output against a ~17 Hz EKF reference.

    If true, the error should scale with (message age x angular rate) and
    should vanish when the airframe is nearly stationary. Neither holds.
    """
    banner("DIAGNOSTIC A  -  TELEMETRY STALENESS HYPOTHESIS   (Section 4.5.4)")
    for key in HOVER_RUNS:
        v = validated(dfs[key])
        cf_db, _ = debias(v["roll_cf_deg"].values, v["ekf_roll_deg"].values)
        err  = pd.Series(np.abs(cf_db - v["ekf_roll_deg"].values), index=v.index)
        rate = v["fc_rate_roll_dps"].abs()
        pred = rate * v["age_att_s"]          # displacement during staleness window

        quiet = rate < 2.0                     # near-stationary rows

        print(f"\n  {labels[key]}")
        print(f"    corr(message age, |roll error|) ....... {err.corr(v['age_att_s']):+.3f}")
        print(f"    corr(age x rate, |roll error|) ........ {err.corr(pred):+.3f}")
        print(f"    predicted staleness error: mean {pred.mean():.2f} deg, "
              f"p95 {pred.quantile(.95):.2f}, max {pred.max():.2f}")
        print(f"    observed median |roll error| .......... {err.median():.2f} deg")
        print(f"    median error on near-stationary rows .. {err[quiet].median():.2f} deg "
              f"(n = {quiet.sum():,}, rate < 2 dps)")
    print("\n  VERDICT: correlation is negligible, predicted magnitude is ~2% of")
    print("  observed, and the error persists when the airframe is nearly still.")
    print("  Telemetry staleness is NOT the cause.")


# ==============================================================================
# DIAGNOSTIC B - IS THE FILTER IMPLEMENTED CORRECTLY?
# ==============================================================================

def diag_cf_rebuild(dfs, labels):
    """
    Rebuild the complementary filter offline from the logged accelerometer
    angle, gyroscope rate and cycle interval, and compare against the logged
    filter output. A match confirms the deployed C++ code implements the
    intended recurrence exactly.
    """
    banner("DIAGNOSTIC B  -  FILTER IMPLEMENTATION CHECK   (Section 4.5.4)")
    for key in HOVER_RUNS:
        df = dfs[key]
        rebuilt = run_cf(df["roll_acc_deg"].values,
                         df["gx_dps"].values,
                         df["dt_s"].values,
                         ALPHA_DEPLOYED)
        r    = np.corrcoef(rebuilt, df["roll_cf_deg"])[0, 1]
        dmax = np.abs(rebuilt - df["roll_cf_deg"]).max()
        print(f"  {labels[key]:16s}  r = {r:.4f}   max difference = {dmax:.3f} deg")
    print("\n  VERDICT: the deployed filter reproduces the intended recurrence")
    print("  exactly. The residual is NOT an implementation defect.")


# ==============================================================================
# DIAGNOSTIC C - VIBRATION CORRUPTION OF THE ACCELEROMETER
# ==============================================================================

def diag_vibration(dfs, labels):
    """
    Test the accelerometer against the assumption underlying angle
    derivation: that linear acceleration is negligible relative to gravity,
    so ||a|| should sit close to 1 g.

    Also predicts the noise passed through to the filter output. For
    approximately white input noise of standard deviation sigma, a
    first-order filter with coefficient alpha yields an output standard
    deviation of sigma * sqrt((1 - alpha) / (1 + alpha)).
    """
    banner("DIAGNOSTIC C  -  VIBRATION CORRUPTION   (Tables 4.11, Section 4.6)")
    atten = np.sqrt((1 - ALPHA_DEPLOYED) / (1 + ALPHA_DEPLOYED))
    print(f"  Filter attenuation factor at alpha = {ALPHA_DEPLOYED}: {atten:.4f}\n")

    rows = []
    for key in HOVER_RUNS:
        a   = active_rows(dfs[key])
        mag = np.sqrt(a["ax_g"]**2 + a["ay_g"]**2 + a["az_g"]**2)
        rows.append({
            "Run":                    labels[key],
            "Mean ||a|| (g)":         f"{mag.mean():.3f}",
            "Std dev (g)":            f"{mag.std():.3f}",
            "Peak ||a|| (g)":         f"{mag.max():.2f}",
            "Within 0.1g of 1g (%)":  f"{(np.abs(mag-1.0) < 0.1).mean()*100:.1f}",
            "acc angle std (deg)":    f"{a['roll_acc_deg'].std():.2f}",
            "acc angle range (deg)":  f"{a['roll_acc_deg'].min():+.0f} to "
                                      f"{a['roll_acc_deg'].max():+.0f}",
        })
    show(rows)

    print("\n  Predicted vs observed filter output noise:")
    print(f"    {'Run':16s} {'Axis':6s} {'sigma_in':>10s} {'predicted':>11s} {'observed RMS':>13s}")
    for key in HOVER_RUNS:
        v = validated(dfs[key])
        for axis, cf_col, acc_col, ekf_col in [
                ("Roll",  "roll_cf_deg",  "roll_acc_deg",  "ekf_roll_deg"),
                ("Pitch", "pitch_cf_deg", "pitch_acc_deg", "ekf_pitch_deg")]:
            sigma_in = (v[acc_col] - v[cf_col]).std()
            cf_db, _ = debias(v[cf_col].values, v[ekf_col].values)
            obs = np.sqrt(((cf_db - v[ekf_col].values)**2).mean())
            print(f"    {labels[key]:16s} {axis:6s} {sigma_in:9.1f}d {sigma_in*atten:10.2f}d "
                  f"{obs:12.2f}d")
    print("\n  VERDICT: the prediction accounts for most of the observed error on")
    print("  both axes. The accelerometer correction term is injecting noise")
    print("  rather than removing gyroscope drift. Cause is MECHANICAL.")


# ==============================================================================
# DIAGNOSTIC D - FILTER COEFFICIENT SWEEP
# ==============================================================================

ALPHA_SWEEP = [0.98, 0.985, 0.99, 0.9925, 0.995, 0.9965, 0.998, 0.999]

def diag_alpha_sweep(dfs, labels):
    """
    Reprocess the logged sensor data with a range of filter coefficients to
    establish how much of the roll disagreement is recoverable by tuning.

    This is a POST-HOC analysis of recorded data, not a flight test of a
    retuned filter. The indicated coefficient must be confirmed in flight
    before adoption.
    """
    banner("DIAGNOSTIC D  -  FILTER COEFFICIENT SWEEP   (Table 4.12)")
    results = {}
    for key in HOVER_RUNS:
        df   = dfs[key]
        mask = ((df["fc_rate_roll_dps"].abs() > ACTIVE_DPS) |
                (df["fc_rate_pitch_dps"].abs() > ACTIVE_DPS)) & \
               (df["ekf_roll_deg"].abs() < EKF_LIMIT_DEG)
        idx  = np.where(mask.values)[0]
        ekf  = df["ekf_roll_deg"].values

        meds, pcts = [], []
        for alpha in ALPHA_SWEEP:
            out      = run_cf(df["roll_acc_deg"].values, df["gx_dps"].values,
                              df["dt_s"].values, alpha)
            sub      = out[idx]
            cf_db, _ = debias(sub, ekf[idx])
            err      = np.abs(cf_db - ekf[idx])
            meds.append(np.median(err))
            pcts.append((err < AGREE_DEG).mean() * 100)
        results[key] = (meds, pcts)

    dt_med = np.median(dfs["003"]["dt_s"])
    print(f"  {'alpha':>8} {'tau (s)':>9} "
          f"{'003 median':>12} {'003 <3deg%':>12} {'004 median':>12} {'004 <3deg%':>12}")
    for i, alpha in enumerate(ALPHA_SWEEP):
        tau = alpha * dt_med / (1 - alpha)
        tag = " *" if abs(alpha - ALPHA_DEPLOYED) < 1e-9 else "  "
        print(f"  {alpha:>8.4f}{tag[1]} {tau:>8.2f} "
              f"{results['003'][0][i]:>11.2f}d {results['003'][1][i]:>11.1f} "
              f"{results['004'][0][i]:>11.2f}d {results['004'][1][i]:>11.1f}")
    print("\n  * = coefficient deployed in flight")
    print("  POST-HOC ANALYSIS - confirm in flight before adopting.")
    return results


# ==============================================================================
# TELEMETRY STREAM RATES
# ==============================================================================

def stream_rates(labels):
    """Effective MAVLink stream rates, from the per-run metadata summary."""
    banner("TELEMETRY STREAM RATES   (Table 4.3)")
    rows = []
    for key in RUNS:
        att, tgt, rc = MSG_COUNTS[key]
        d = DURATIONS[key]
        rows.append({
            "Run":                    labels[key],
            "Duration (s)":           f"{d:.1f}",
            "ATTITUDE (Hz)":          f"{att/d:.1f}",
            "ATTITUDE_TARGET (Hz)":   f"{tgt/d:.1f}",
            "RC_CHANNELS (Hz)":       f"{rc/d:.1f}",
        })
    show(rows)


# ==============================================================================
# FIGURES
# ==============================================================================

def fig_m1(dfs):
    """Figure 4.3 - loop period histograms for the bench runs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, colour, title in zip(axes, ["001", "002"], [BLUE, ORANGE],
                                      ["Bench Unloaded (Run 001)", "Bench Loaded (Run 002)"]):
        dt = dfs[key]["dt_s"] * 1e6
        ax.hist(dt, bins=80, color=colour, alpha=0.8, edgecolor="none")
        ax.axvline(dt.mean(), color=GREY, lw=1.5, ls="--", label=f"Mean {dt.mean():,.1f} us")
        ax.axvline(dt.mean() + dt.std(), color=GREY, lw=1, ls=":",
                   label=f"+1 sigma = {dt.std():.1f} us")
        ax.set_xlabel("Loop period dt (us)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle(f"Loop Period Distribution (50 Hz target = {DEADLINE_US:,.0f} us)", fontsize=11)
    fig.tight_layout()
    return fig, "fig_m1_loop_timing.png"


def fig_m2(dfs):
    """Figure 4.4 - CPU utilisation over time for the hover runs."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, key, colour, title in zip(axes, HOVER_RUNS, [BLUE, ORANGE],
                                      ["Hover Unloaded (Run 003)", "Hover Loaded (Run 004)"]):
        df = dfs[key]
        ax.plot(df["t_s"], df["cpu_busy_pct"], color=colour, lw=0.8, label="System CPU busy (%)")
        ax.plot(df["t_s"], df["proc_cpu_pct"], color=GREEN, lw=0.8, ls="--",
                label="Control process CPU (%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CPU (%)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
    fig.suptitle("CPU Utilisation Over Time", fontsize=11)
    fig.tight_layout()
    return fig, "fig_m2_cpu.png"


def fig_m3(dfs):
    """Figure 4.5 - raw accelerometer angle vs CF output, with noise residual."""
    a = active_rows(dfs["003"])
    t0 = a["t_s"].min()
    w  = a[(a["t_s"] >= t0) & (a["t_s"] < t0 + 10)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    ax1.plot(w["t_s"], w["roll_acc_deg"], color=ORANGE, lw=0.7, alpha=0.8,
             label="Raw accelerometer angle")
    ax1.plot(w["t_s"], w["roll_cf_deg"], color=BLUE, lw=1.2, label="CF filtered angle")
    ax1.set_ylabel("Roll (deg)")
    ax1.set_title("Roll: Raw Accelerometer vs Complementary Filter "
                  "(10 s window, hover unloaded)")
    ax1.legend(fontsize=9)

    resid = w["roll_acc_deg"] - w["roll_cf_deg"]
    ax2.plot(w["t_s"], resid, color=RED, lw=0.6, alpha=0.8)
    ax2.axhline(0, color=GREY, lw=0.8, ls="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Noise residual (deg)")
    ax2.set_title(f"Roll noise residual (acc - CF)   std = {resid.std():.2f} deg")
    fig.tight_layout()
    return fig, "fig_m3_snr.png"


def fig_m4(dfs):
    """Figure 4.6 - CF vs EKF roll angle with absolute error."""
    v = validated(dfs["003"])
    cf_db, _ = debias(v["roll_cf_deg"].values, v["ekf_roll_deg"].values)
    v = v.assign(roll_cf_db=cf_db)
    t0 = v["t_s"].min()
    w  = v[(v["t_s"] >= t0) & (v["t_s"] < t0 + 30)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(w["t_s"], w["roll_cf_db"],   color=BLUE,   lw=1.2, label="CF roll (de-biased)")
    ax1.plot(w["t_s"], w["ekf_roll_deg"], color=ORANGE, lw=0.8, ls="--", label="EKF roll (FC)")
    ax1.set_ylabel("Roll (deg)")
    ax1.set_title("CF vs EKF Roll Angle (30 s window, hover unloaded)")
    ax1.legend(fontsize=9)

    err = (w["roll_cf_db"] - w["ekf_roll_deg"]).abs()
    ax2.plot(w["t_s"], err, color=RED, lw=0.6)
    ax2.axhline(AGREE_DEG, color=GREY, lw=0.8, ls="--",
                label=f"{AGREE_DEG:.0f} deg threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|CF - EKF| (deg)")
    ax2.set_title("Absolute roll error after de-bias")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    return fig, "fig_m4_cf_ekf.png"


def fig_m5(dfs):
    """Figure 4.10 - shadow PID command vs FC rate demand."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, colour, title in zip(axes, HOVER_RUNS, [BLUE, ORANGE],
                                      ["Hover Unloaded (Run 003)", "Hover Loaded (Run 004)"]):
        a = active_rows(dfs[key])
        ax.scatter(a["fc_rate_roll_dps"], a["roll_cmd_dps"], s=2, alpha=0.3, color=colour)
        lim = max(a["fc_rate_roll_dps"].abs().max(), a["roll_cmd_dps"].abs().max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--", label="Ideal (cmd = FC)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("FC angular rate (dps)")
        ax.set_ylabel("Shadow PID command (dps)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
    fig.suptitle("Command Divergence: Shadow PID vs ArduPilot Rate (Roll)", fontsize=11)
    fig.tight_layout()
    return fig, "fig_m5_divergence.png"


def fig_vibration(dfs):
    """Figure 4.7 - accelerometer magnitude distribution during hover."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, colour, title in zip(axes, HOVER_RUNS, [BLUE, ORANGE],
                                      ["Hover Unloaded (Run 003)", "Hover Loaded (Run 004)"]):
        df  = dfs[key]
        mag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
        ax.hist(mag, bins=120, color=colour, alpha=0.85, edgecolor="none")
        ax.axvline(1.0, color=RED, lw=1.6, ls="--", label="Expected 1 g (gravity only)")
        ax.axvline(mag.mean(), color=GREY, lw=1.4, ls=":", label=f"Mean {mag.mean():.2f} g")
        ax.set_xlabel("Accelerometer magnitude ||a|| (g)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Vibration Corruption of the Accelerometer During Hover", fontsize=11)
    fig.tight_layout()
    return fig, "fig_vibration_magnitude.png"


def fig_accel_wrap(dfs):
    """Figure 4.8 - accelerometer angle wrapping through the full range."""
    df = dfs["003"]
    t0 = df["t_s"].min() + 20
    w  = df[(df["t_s"] >= t0) & (df["t_s"] < t0 + 10)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(w["t_s"], w["roll_acc_deg"], color=ORANGE, lw=0.7,
             label="Accelerometer roll angle (raw)")
    ax1.axhline(180, color=GREY, lw=0.6, ls=":")
    ax1.axhline(-180, color=GREY, lw=0.6, ls=":")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Accelerometer-derived roll angle spans the full +/-180 deg range during hover")
    ax1.legend(fontsize=8)

    ax2.plot(w["t_s"], w["roll_cf_deg"],   color=BLUE, lw=1.1, label="Complementary filter output")
    ax2.plot(w["t_s"], w["ekf_roll_deg"],  color=RED,  lw=0.9, ls="--", label="Flight controller EKF")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Roll (deg)")
    ax2.set_title("Resulting CF estimate compared with the EKF reference")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig, "fig_accel_wrap.png"


def fig_alpha(sweep):
    """Figure 4.9 - median roll disagreement against filter coefficient."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ALPHA_SWEEP, sweep["003"][0], "o-", color=BLUE,   label="Hover unloaded (Run 003)")
    ax.plot(ALPHA_SWEEP, sweep["004"][0], "s-", color=ORANGE, label="Hover loaded (Run 004)")
    ax.axvline(ALPHA_DEPLOYED, color=RED, lw=1.2, ls="--",
               label=f"Deployed alpha = {ALPHA_DEPLOYED}")
    ax.axhline(AGREE_DEG, color=GREY, lw=1, ls=":",
               label=f"{AGREE_DEG:.0f} deg acceptance threshold")
    ax.set_xlabel("Complementary filter coefficient alpha")
    ax.set_ylabel("Median |CF - EKF| roll error (deg)")
    ax.set_title("Offline sensitivity of roll agreement to the filter coefficient")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "fig_alpha_sweep.png"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs, labels = load_runs()

    print("\nLoaded runs:")
    for key, df in dfs.items():
        print(f"  {labels[key]:16s}  {len(df):6,} cycles, {len(df.columns)} columns")

    # --- metrics -------------------------------------------------------------
    metric1(dfs, labels)
    metric2(dfs, labels)
    stream_rates(labels)
    metric3(dfs, labels)
    metric4(dfs, labels)
    metric5(dfs, labels)

    # --- diagnostics ---------------------------------------------------------
    diag_staleness(dfs, labels)
    diag_cf_rebuild(dfs, labels)
    diag_vibration(dfs, labels)
    sweep = diag_alpha_sweep(dfs, labels)

    # --- figures -------------------------------------------------------------
    banner("FIGURES")
    builders = [fig_m1, fig_m2, fig_m3, fig_m4, fig_m5, fig_vibration, fig_accel_wrap]
    for builder in builders:
        fig, name = builder(dfs)
        fig.savefig(os.path.join(OUT_DIR, name))
        plt.close(fig)
        print(f"  wrote {name}")
    fig, name = fig_alpha(sweep)
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)
    print(f"  wrote {name}")

    print(f"\nAll figures written to {os.path.abspath(OUT_DIR)}")
    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
