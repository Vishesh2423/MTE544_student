"""
plot_ekf.py - Report-ready plots for MTE 544 Lab 3 (EKF localization).

For each logged CSV it produces one figure comparing EKF estimates against
the raw measurements (IMU, and odometry if logged), with titles, axis labels,
legends, distinct colours/markers and grids, and saves it as a PNG.
It also prints the RMS difference (measurement - estimate) per channel so you
can compare the Q/R variations quantitatively.

Usage:
    python3 plot_ekf.py --files spiral_Q05_R05.csv spiral_Q5_R05.csv ...
    python3 plot_ekf.py --files a.csv b.csv --labels "Q=0.5,R=0.5" "Q=5,R=0.5"
    python3 plot_ekf.py --files *.csv --no-show          # just save PNGs

Columns used (whitespace and the trailing empty column are handled):
    required: kf_x, kf_y, kf_vx, kf_w, kf_ax, kf_ay, imu_ax, imu_ay, stamp
    optional: odom_v, odom_w   -> plotted against kf_vx / kf_w if present
              odom_x, odom_y   -> plotted against kf_x / kf_y if present
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

# Measurement -> estimate pairs. Each is plotted on its own axis.
PAIRS = [
    ("imu_ax", "kf_ax", r"$a_x$ [m/s$^2$]", "Linear acceleration $a_x$"),
    ("imu_ay", "kf_ay", r"$a_y$ [m/s$^2$]", "Linear acceleration $a_y$"),
    ("odom_v", "kf_vx", r"$v$ [m/s]", "Linear velocity $v$"),
    ("odom_w", "kf_w", r"$\omega$ [rad/s]", r"Angular velocity $\omega$"),
]
MEAS_STYLE = dict(color="tab:gray", linestyle="-", marker=".", markersize=3,
                  linewidth=0.8, alpha=0.7)
EST_STYLE = dict(color="tab:red", linestyle="--", linewidth=1.8)


def load(filename):
    """Read the logged CSV into a dict of numpy arrays + time in seconds."""
    with open(filename, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    headers = [h.strip() for h in rows[0]]
    keep = [i for i, h in enumerate(headers) if h]          # drop empty column
    data = {headers[i]: np.array([float(r[i]) for r in rows[1:]]) for i in keep}
    stamp = data["stamp"]
    # ROS stamps are in ns; fall back to s if they are already small
    scale = 1e-9 if stamp[0] > 1e12 else 1.0
    data["t"] = (stamp - stamp[0]) * scale
    return data


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_run(data, label, outdir, dpi):
    t = data["t"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"EKF estimate vs. measurement - {label}", fontsize=14)
    stats = {}

    # (1) Trajectory in the x-y plane
    ax = axes[0, 0]
    if "odom_x" in data and "odom_y" in data:
        ax.plot(data["odom_x"], data["odom_y"], label="Odometry (measured)",
                **MEAS_STYLE)
    ax.plot(data["kf_x"], data["kf_y"], label="EKF estimate", **EST_STYLE)
    ax.plot(data["kf_x"][0], data["kf_y"][0], "go", label="Start")
    ax.plot(data["kf_x"][-1], data["kf_y"][-1], "ks", label="End")
    ax.set(title="Robot trajectory (x-y)", xlabel="x [m]", ylabel="y [m]")
    ax.axis("equal")

    # (2) x and y position over time
    ax = axes[0, 1]
    ax.plot(t, data["kf_x"], color="tab:blue", label="EKF x")
    ax.plot(t, data["kf_y"], color="tab:orange", linestyle="-.", label="EKF y")
    if "odom_x" in data:
        ax.plot(t, data["odom_x"], color="tab:blue", marker=".", markersize=2,
                linestyle="none", alpha=0.5, label="Odom x")
        ax.plot(t, data["odom_y"], color="tab:orange", marker=".", markersize=2,
                linestyle="none", alpha=0.5, label="Odom y")
    ax.set(title="Position vs. time", xlabel="Time [s]", ylabel="Position [m]")

    # (3-6) measurement vs estimate for each measured state
    for ax, (meas, est, ylab, title) in zip(axes.flat[2:], PAIRS):
        if meas in data:
            ax.plot(t, data[meas], label=f"Measured ({meas})", **MEAS_STYLE)
            stats[est] = rms(data[meas], data[est])
            title += f"  (RMS diff = {stats[est]:.3f})"
        ax.plot(t, data[est], label=f"EKF estimate ({est})", **EST_STYLE)
        ax.set(title=title, xlabel="Time [s]", ylabel=ylab)

    for ax in axes.flat:
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    path = os.path.join(outdir, f"{safe}.png")
    fig.savefig(path, dpi=dpi)
    print(f"saved {path}")
    return stats


def main():
    p = argparse.ArgumentParser(description="Plot EKF logs for Lab 3.")
    p.add_argument("--files", nargs="+", required=True, help="CSV log files")
    p.add_argument("--labels", nargs="+",
                   help="Plot labels, e.g. 'Q=0.5, R=0.5' (default: file name)")
    p.add_argument("--outdir", default="plots", help="Where PNGs are saved")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-show", action="store_true", help="Only save PNGs")
    args = p.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(f))[0]
                             for f in args.files]
    if len(labels) != len(args.files):
        p.error("--labels must have one entry per file")
    os.makedirs(args.outdir, exist_ok=True)

    summary = {}
    for f, lab in zip(args.files, labels):
        summary[lab] = plot_run(load(f), lab, args.outdir, args.dpi)

    # RMS table for the report (measurement - estimate)
    cols = [est for _, est, _, _ in PAIRS]
    print("\nRMS(measurement - estimate):")
    print(f"{'run':<25}" + "".join(f"{c:>10}" for c in cols))
    for lab, s in summary.items():
        print(f"{lab:<25}" + "".join(
            f"{s[c]:>10.4f}" if c in s else f"{'-':>10}" for c in cols))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
