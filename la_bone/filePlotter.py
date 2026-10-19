# Plots the logged IMU, odometry and laser data as a 3x3 grid
# rows: imu, odom, laser   columns: line, circle, spiral

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import FileReader

MOTIONS = ["line", "circle", "spiral"]
# Beam 0 is not along the robot's heading. Searching offsets for the one that makes
# repeated scans of the same walls overlap best gave -90 deg for all three runs.
LASER_YAW_OFFSET = -np.pi / 2
LASER_TIME_OFFSET = 0.0  # s added to laser stamps to line them up with odom/imu
STAMP_REF = 0.0          # beam 0 time relative to the stamp, in scan periods
# Direction in which time advances with beam index. Searching offset / stamp
# convention / direction for the best scan overlap picked -1 (beam 0 is taken
# first and later beams EARLIER in time, i.e. the stamp is the end of the sweep);
# with that sign no extra time offset was needed. +1 made the overlap worse.
SWEEP_DIR = -1
SMOOTH_SAMPLES = 9      # odom samples averaged when estimating velocity
IMU_SMOOTH_SAMPLES = 5  # imu samples averaged when estimating yaw rate
MOTION_TITLES = {"line": "Straight Line", "circle": "Circle", "spiral": "Spiral"}


def read_columns(filename):
    headers, values = FileReader(filename).read_file()
    cols = {h: np.array([row[i] for row in values]) for i, h in enumerate(headers)}
    cols["t"] = (cols["stamp"] - cols["stamp"][0]) * 1e-9  # seconds since start
    return cols


def read_laser_frame(filename, frame=None):
    """Return (ranges, angle_increment) for one scan. frame=None -> middle scan."""
    with open(filename) as f:
        lines = f.read().strip().split("\n")[1:]
    if frame is None:
        frame = len(lines) // 2
    line = lines[frame]
    inside = line[line.index("[") + 1:line.index("]")]
    ranges = np.array([float(v) for v in inside.split(",")])
    angle_inc = float(line[line.index("])") + 2:].split(",")[1])
    return ranges, angle_inc, frame


def plot_imu(ax, motion):
    d = read_columns(f"imu_content_{motion}.csv")
    ax.plot(d["t"], d["acc_x"], label="acc_x [m/s²]")
    ax.plot(d["t"], d["acc_y"], label="acc_y [m/s²]")
    ax.plot(d["t"], d["angular_z"], label="angular_z [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration / angular rate")
    ax.set_title(f"IMU - {MOTION_TITLES[motion]}")


def plot_odom(ax, motion):
    d = read_columns(f"odom_content_{motion}.csv")
    ax.plot(d["x"], d["y"], label="path")
    ax.plot(d["x"][0], d["y"][0], "go", label="start")
    ax.plot(d["x"][-1], d["y"][-1], "rs", label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Odometry - {MOTION_TITLES[motion]}")
    ax.axis("equal")


def plot_laser(ax, motion):
    ranges, inc, frame = read_laser_frame(f"laser_content_{motion}.csv")
    angles = np.arange(len(ranges)) * inc
    ok = np.isfinite(ranges)
    ax.scatter(angles[ok], ranges[ok], s=6, label=f"ranges (scan #{frame})")
    ax.set_theta_zero_location("E")  # 0 rad along +x
    ax.set_theta_direction(1)        # angle increases counter-clockwise
    ax.set_xlabel("Beam angle [deg, CCW from +x]")
    ax.set_ylabel("Range [m]", labelpad=25)
    ax.set_title(f"Laser scan - {MOTION_TITLES[motion]}")


def plot_all(save=None):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for col, motion in enumerate(MOTIONS):
        plot_imu(axes[0][col], motion)
        plot_odom(axes[1][col], motion)
        axes[2][col].remove()  # replace with a polar axis
        axes[2][col] = fig.add_subplot(3, 3, 7 + col, projection="polar")
        plot_laser(axes[2][col], motion)
    for ax in axes.flat:
        ax.grid(True)
        ax.legend(fontsize=8, **(dict(loc="upper right", bbox_to_anchor=(1.25, 1.12)) if ax.name == "polar" else dict(loc="best")))
    fig.suptitle("IMU, Odometry and Laser Data (rows) for Line, Circle, Spiral (columns)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()


def read_all_laser(filename):
    """Return (stamps [s], ranges [N x beams], angle_increment) for every scan."""
    with open(filename) as f:
        lines = f.read().strip().split("\n")[1:]
    stamps, scans = [], []
    for line in lines:
        inside = line[line.index("[") + 1:line.index("]")]
        scans.append([float(v) for v in inside.split(",")])
        tail = line[line.index("])") + 2:].split(",")
        inc = float(tail[1])
        stamps.append(float(tail[2]))
    return np.array(stamps), np.array(scans), inc


def read_map(yaml_file="map.yaml"):
    """Load the occupancy map image and return (image, extent) in world metres."""
    cfg = {}
    with open(yaml_file) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
    res = float(cfg["resolution"])
    ox, oy, _ = [float(v) for v in cfg["origin"].strip("[]").split(",")]
    img = plt.imread(cfg["image"])
    h, w = img.shape[:2]
    return img, (ox, ox + w * res, oy, oy + h * res)


def smooth(v, n):
    return np.convolve(v, np.ones(n) / n, mode="same")


def project_scans(motion, deskew=False, time_offset=None, stamp_ref=None, sweep_dir=None):
    """Project every laser scan into the odometry frame.
    Returns (px, py, t, valid, odom, n_scans). With deskew=True each beam is
    corrected for the robot motion during the scan sweep: beam k was taken
    k*scan_dt/n_beams after the scan stamp, so its pose is the odom pose at the
    stamp advanced by that delay -- heading by the IMU yaw rate (angular_z),
    position by the odom velocity."""
    odom = read_columns(f"odom_content_{motion}.csv")
    stamps, scans, inc = read_all_laser(f"laser_content_{motion}.csv")
    n_beams = scans.shape[1]
    t0 = odom["stamp"][0]
    time_offset = LASER_TIME_OFFSET if time_offset is None else time_offset
    stamp_ref = STAMP_REF if stamp_ref is None else stamp_ref
    sweep_dir = SWEEP_DIR if sweep_dir is None else sweep_dir
    t_scan = (stamps - t0) * 1e-9 + time_offset  # laser time -> odom/imu clock
    th_unw = np.unwrap(odom["th"])
    x = np.interp(t_scan, odom["t"], odom["x"])
    y = np.interp(t_scan, odom["t"], odom["y"])
    th = np.interp(t_scan, odom["t"], th_unw)
    beam = np.arange(n_beams) * inc
    ang = th[:, None] + LASER_YAW_OFFSET + beam[None, :]
    px0, py0 = x[:, None], y[:, None]
    if deskew:
        scan_dt = np.median(np.diff(t_scan))  # scans are back to back
        delay = (sweep_dir * np.arange(n_beams) * scan_dt / n_beams + stamp_ref * scan_dt)[None, :]
        # yaw rate from the IMU (smoothed, it is noisy)
        imu = read_columns(f"imu_content_{motion}.csv")
        t_imu = (imu["stamp"] - t0) * 1e-9
        omega = np.interp(t_scan, t_imu, smooth(imu["angular_z"], IMU_SMOOTH_SAMPLES))
        ang = ang + omega[:, None] * delay
        # linear velocity in the odom frame from odom positions (published in steps)
        vx = np.interp(t_scan, odom["t"], smooth(np.gradient(odom["x"], odom["t"]), SMOOTH_SAMPLES))
        vy = np.interp(t_scan, odom["t"], smooth(np.gradient(odom["y"], odom["t"]), SMOOTH_SAMPLES))
        px0 = px0 + vx[:, None] * delay
        py0 = py0 + vy[:, None] * delay
    px = px0 + scans * np.cos(ang)
    py = py0 + scans * np.sin(ang)
    t = np.broadcast_to(t_scan[:, None], scans.shape)
    return px, py, t, np.isfinite(scans), odom, len(scans)


def plot_laser_map(save=None):
    """2x3 figure: all scans of each run in the odom frame, coloured by time.
    Top row uses the pose at the scan stamp; bottom row corrects each scan for
    the robot motion (IMU yaw rate + odom velocity) during its sweep."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 13))
    for row, deskew in enumerate([False, True]):
        for ax, motion in zip(axes[row], MOTIONS):
            px, py, t, ok, odom, n_scans = project_scans(motion, deskew)
            sc = ax.scatter(px[ok], py[ok], c=t[ok], s=1.5, cmap="viridis",
                            label=f"laser hits ({n_scans} scans)")
            ax.plot(odom["x"], odom["y"], "r-", lw=1.5, label="odom path")
            ax.plot(odom["x"][0], odom["y"][0], "go", label="start")
            ax.plot(odom["x"][-1], odom["y"][-1], "rs", label="end")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Scan time [s]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            kind = "motion-corrected" if deskew else "uncorrected"
            ax.set_title(f"All laser scans ({kind}) - {MOTION_TITLES[motion]}")
            ax.grid(True)
            leg = ax.legend(fontsize=8, loc="upper right")
            leg.legendHandles[0].set_sizes([20])
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3x3 plot of imu/odom/laser data")
    parser.add_argument("--save", default=None, help="optional output image path")
    parser.add_argument("--laser-map", default=None,
                        help="also save the 1x3 all-scans world-frame laser plot here")
    args = parser.parse_args()
    if args.laser_map:
        plot_laser_map(args.laser_map)
    plot_all(args.save)
