"""
plot_errors.py - plot and sanity-check the CSV logs from the TurtleBot controller.

Reads (per run folder):
    linear.csv       e, e_dot, e_int, stamp     (distance to goal [m])
    angular.csv      e, e_dot, e_int, stamp     (heading error [rad])
    robot_pose.csv   x, y, theta, stamp         (optional, for x-t / y-t / th-t / x-y)

Usage (from the folder containing the csv files):
    python3 plot_errors.py                                   # point run, current folder
    python3 plot_errors.py --goal -1 -1                      # also check logged error vs. pose
    python3 plot_errors.py runs/P runs/PID --goal -1 -1      # overlay two runs (P vs PID)
    python3 plot_errors.py runs/parab --traj parabola        # trajectory run
    python3 plot_errors.py --no-show                         # just save the PNGs

Saves: errors_vs_time.png, phase_e_edot.png, pose.png (in the first run folder).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_csv(path):
    """Read a log written by utilities.Logger -> dict {column name: np.array}.

    Own reader on purpose: utilities.FileReader calls next(file) after the header,
    which silently throws away the first data row.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    headers = [h.strip() for h in lines[0].split(",") if h.strip()]
    rows = []
    for ln in lines[1:]:
        vals = [v.strip() for v in ln.split(",") if v.strip()]
        if len(vals) == len(headers):          # skip half-written last line
            rows.append([float(v) for v in vals])
    data = np.array(rows)
    out = {h: data[:, i] for i, h in enumerate(headers)}
    out["t"] = (out["stamp"] - out["stamp"][0]) / 1e9   # stamp is in ns -> seconds
    return out


def load_run(folder):
    run = {"name": os.path.basename(os.path.abspath(folder)) or folder}
    for key, fname in [("lin", "linear.csv"), ("ang", "angular.csv"), ("pose", "robot_pose.csv")]:
        path = os.path.join(folder, fname)
        run[key] = load_csv(path) if os.path.exists(path) else None
        if run[key] is None:
            print(f"[{run['name']}] note: {fname} not found, skipping")
    return run


# ----------------------------------------------------------------------------
# Logging sanity checks
# ----------------------------------------------------------------------------
def check_log(name, label, d):
    """Print warnings if the log looks wrong. Returns number of warnings."""
    warn = 0
    t = d["t"]
    dt = np.diff(t)
    if len(t) < 5:
        print(f"[{name}] {label}: only {len(t)} rows logged, nothing to check")
        return 1
    med = np.median(dt[dt > 0]) if np.any(dt > 0) else float("nan")
    print(f"[{name}] {label}: {len(t)} rows, {t[-1]:.1f} s, median dt {med*1000:.0f} ms (~{1/med:.1f} Hz)")

    n_dup = int(np.sum(dt == 0))
    if n_dup:
        warn += 1
        print(f"   WARNING {n_dup} repeated timestamps ({100*n_dup/len(dt):.0f}% of rows): the controller ran "
              f"without a new /odom message, so e_dot is computed with dt=0 (falls back to 0.1 s and a 0 "
              f"difference). Expect e_dot spikes/dropouts.")
    if np.any(dt < 0):
        warn += 1
        print("   WARNING timestamps go backwards")
    if np.any(dt > 3 * med):
        warn += 1
        print(f"   WARNING {int(np.sum(dt > 3*med))} gaps > 3x median dt (max {dt.max():.2f} s)")
    if np.any(~np.isfinite(d["e"])) or np.any(~np.isfinite(d["e_dot"])):
        warn += 1
        print("   WARNING NaN/inf values in log")

    # Is the logged e_dot the average slope over the 3-sample window used in pid.py?
    e, ed = d["e"], d["e_dot"]
    ok = dt[:-1] + dt[1:] > 0
    ref = (e[2:] - e[:-2]) / np.where(ok, t[2:] - t[:-2], np.nan)
    m = np.isfinite(ref) & np.isfinite(ed[2:])
    if m.sum() > 5 and np.sum(ref[m] ** 2) > 0:
        ratio = np.sum(ed[2:][m] * ref[m]) / np.sum(ref[m] ** 2)
        if abs(ratio - 1) > 0.1:
            warn += 1
            print(f"   WARNING logged e_dot is ~{ratio:.2f}x the slope of the logged e. In pid.py, "
                  f"error_dot and dt_avg are divided by len(history) but the loop only adds len(history)-1 "
                  f"terms -> divide by len(history)-1.")
    return warn


def check_against_pose(run, goal, point_mode):
    """Recompute errors from robot_pose.csv + goal and compare with what the PID logged."""
    pose, lin, ang = run["pose"], run["lin"], run["ang"]
    if pose is None or goal is None:
        return
    name = run["name"]
    gx, gy = goal
    for label, d in [("linear", lin), ("angular", ang)]:
        if d is None:
            continue
        if label == "angular" and not point_mode:
            continue   # trajectory mode steers at a look-ahead point, not the final goal
        x = np.interp(d["stamp"], pose["stamp"], pose["x"])
        y = np.interp(d["stamp"], pose["stamp"], pose["y"])
        th = np.interp(d["stamp"], pose["stamp"], np.unwrap(pose["theta"]))
        if label == "linear":
            expect = np.hypot(gx - x, gy - y)
        else:
            raw = np.arctan2(gy - y, gx - x) - th
            expect = np.arctan2(np.sin(raw), np.cos(raw))
        err = np.abs(expect - d["e"])
        if np.max(err) > 0.05:
            print(f"[{name}] {label}: WARNING logged e differs from error recomputed from robot_pose.csv "
                  f"(max diff {np.max(err):.3f}). Wrong --goal, or the error is being logged from a "
                  f"different pose/stamp than the pose log.")
        else:
            print(f"[{name}] {label}: logged e matches pose + goal (max diff {np.max(err):.4f})")


def summary(run, tol):
    lin, ang = run["lin"], run["ang"]
    line = f"[{run['name']}]"
    if lin is not None:
        reached = np.where(lin["e"] < tol)[0]
        t_reach = f"{lin['t'][reached[0]]:.2f} s" if len(reached) else "never"
        line += f" time to <{tol} m: {t_reach} | final lin err {lin['e'][-1]:.3f} m"
    if ang is not None:
        e = ang["e"]
        cross = np.where(np.sign(e[1:]) != np.sign(e[:-1]))[0]
        if len(cross):
            ov = np.max(np.abs(e[cross[0] + 1:])) if cross[0] + 1 < len(e) else 0
            line += f" | heading: crosses 0 at {ang['t'][cross[0]]:.2f} s, peak after = {np.degrees(ov):.1f} deg"
        else:
            line += " | heading never crosses 0"
    print(line)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def style(ax, title, xlabel, ylabel, zero=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    if zero:
        ax.axhline(0, color="k", lw=0.8)


def fig_errors_vs_time(runs, tol, show_int):
    cols = 3 if show_int else 2
    fig, axes = plt.subplots(2, cols, figsize=(5.5 * cols, 7), sharex="col", squeeze=False)
    spec = [("lin", "Linear (distance to goal)", "m", "m/s", "m*s"),
            ("ang", "Angular (heading error)", "rad", "rad/s", "rad*s")]
    for r, (key, title, u_e, u_d, u_i) in enumerate(spec):
        for run in runs:
            d = run[key]
            if d is None:
                continue
            axes[r, 0].plot(d["t"], d["e"], label=run["name"])
            axes[r, 1].plot(d["t"], d["e_dot"], label=run["name"])
            if show_int:
                axes[r, 2].plot(d["t"], d["e_int"], label=run["name"])
        style(axes[r, 0], f"{title}: error", "time [s]", f"error [{u_e}]")
        style(axes[r, 1], f"{title}: error rate", "time [s]", f"d(error)/dt [{u_d}]")
        if show_int:
            style(axes[r, 2], f"{title}: error integral", "time [s]", f"integral of error [{u_i}]")
        for ax in axes[r]:
            ax.legend()
    axes[0, 0].axhline(tol, color="gray", ls="--", lw=1)
    axes[0, 0].annotate(f"goal tolerance ({tol} m)", (0.98, tol), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="gray")
    fig.suptitle("Controller errors vs time")
    fig.tight_layout()
    return fig


def fig_phase(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    spec = [("lin", "Linear", "error [m]", "d(error)/dt [m/s]"),
            ("ang", "Angular", "error [rad]", "d(error)/dt [rad/s]")]
    for ax, (key, title, xl, yl) in zip(axes, spec):
        for run in runs:
            d = run[key]
            if d is None:
                continue
            line, = ax.plot(d["e"], d["e_dot"], label=run["name"])
            ax.plot(d["e"][0], d["e_dot"][0], "o", color=line.get_color())      # start
            ax.plot(d["e"][-1], d["e_dot"][-1], "s", color=line.get_color())    # end
        style(ax, f"{title}: phase plot (circle = start, square = end)", xl, yl)
        ax.axvline(0, color="k", lw=0.8)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_pose(runs, goal, traj_pts):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_th = fig.add_subplot(gs[2, 1], sharex=ax_x)
    for run in runs:
        p = run["pose"]
        if p is None:
            continue
        line, = ax_xy.plot(p["x"], p["y"], label=run["name"])
        ax_xy.plot(p["x"][0], p["y"][0], "o", color=line.get_color())
        ax_x.plot(p["t"], p["x"], label=run["name"])
        ax_y.plot(p["t"], p["y"], label=run["name"])
        ax_th.plot(p["t"], p["theta"], label=run["name"])
    if traj_pts is not None:
        ax_xy.plot(traj_pts[:, 0], traj_pts[:, 1], "k--", label="reference trajectory")
    if goal is not None:
        ax_xy.plot(*goal, "r*", ms=15, label="goal")
    style(ax_xy, "Robot path (circle = start)", "x [m]", "y [m]", zero=False)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend()
    style(ax_x, "x vs time", "time [s]", "x [m]", zero=False)
    style(ax_y, "y vs time", "time [s]", "y [m]", zero=False)
    style(ax_th, "heading vs time", "time [s]", "theta [rad]", zero=False)
    for ax in (ax_x, ax_y, ax_th):
        ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot and sanity-check controller logs.")
    ap.add_argument("runs", nargs="*", default=["."], help="folder(s) with linear.csv/angular.csv/robot_pose.csv")
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="point goal, enables consistency check")
    ap.add_argument("--traj", choices=["parabola", "sigmoid"], help="plot the reference trajectory from planner.py")
    ap.add_argument("--tol", type=float, default=0.1, help="goal tolerance in m (0.1 point, 0.5 trajectory)")
    ap.add_argument("--int", dest="show_int", action="store_true", help="also plot the error integral")
    ap.add_argument("--no-show", action="store_true", help="save PNGs without opening windows")
    args = ap.parse_args()

    runs = [load_run(f) for f in args.runs]

    traj_pts, goal = None, tuple(args.goal) if args.goal else None
    if args.traj:
        from planner import planner, TRAJECTORY_PLANNER   # planner.py has no ROS imports
        traj_pts = np.array(planner(TRAJECTORY_PLANNER, args.traj).plan())
        goal = tuple(traj_pts[-1])
    point_mode = args.traj is None

    print("\n=== logging checks ===")
    warnings = 0
    for run in runs:
        for label, key in [("linear.csv", "lin"), ("angular.csv", "ang")]:
            if run[key] is not None:
                warnings += check_log(run["name"], label, run[key])
        check_against_pose(run, goal, point_mode)
    print("\n=== summary ===")
    for run in runs:
        summary(run, args.tol)
    print(f"\n{warnings} logging warning(s)\n")

    out = args.runs[0] if os.path.isdir(args.runs[0]) else "."
    figs = {"errors_vs_time.png": fig_errors_vs_time(runs, args.tol, args.show_int),
            "phase_e_edot.png": fig_phase(runs)}
    if any(r["pose"] is not None for r in runs):
        figs["pose.png"] = fig_pose(runs, goal, traj_pts)
    for fname, fig in figs.items():
        fig.savefig(os.path.join(out, fname), dpi=150)
        print("saved", os.path.join(out, fname))
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
