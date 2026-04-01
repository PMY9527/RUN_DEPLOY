#!/usr/bin/env python3
"""Read /cmg_viz_data shared memory and compute per-joint position std (jitter).

Usage:
    python joint_jitter_logger.py                  # print to terminal every 2s
    python joint_jitter_logger.py --csv out.csv    # also save raw samples to CSV
    python joint_jitter_logger.py --window 3.0     # 3-second rolling window
"""
import argparse
import ctypes
import ctypes.util
import mmap
import os
import struct
import sys
import time

import numpy as np

_rt = ctypes.CDLL(ctypes.util.find_library("rt"), use_errno=True)
_rt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
_rt.shm_open.restype = ctypes.c_int

NUM_JOINTS = 29
SHM_NAME = "/cmg_viz_data"

JOINT_NAMES = [
    "left_hip_pitch",    "left_hip_roll",     "left_hip_yaw",
    "left_knee",         "left_ankle_pitch",  "left_ankle_roll",
    "right_hip_pitch",   "right_hip_roll",    "right_hip_yaw",
    "right_knee",        "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",         "waist_roll",        "waist_pitch",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw",
    "left_elbow",        "left_wrist_roll",   "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw",
    "right_elbow",       "right_wrist_roll",  "right_wrist_pitch",
    "right_wrist_yaw",
]

# --- struct layout (must match cmg_viz_shm.h) ---
# atomic<uint32> seq  (4B) + 4B pad + uint64 timestamp_us (8B) = 16B header
HEADER_SIZE = 16
FLOAT_BLOCK = NUM_JOINTS * 4   # 116 bytes per joint array
CMD_BLOCK = 3 * 4              # 12 bytes

# offsets into the mmap
OFF_SEQ        = 0
OFF_TS         = 8
OFF_QREF       = HEADER_SIZE
OFF_QREF_VEL   = OFF_QREF + FLOAT_BLOCK
OFF_ACTUAL_POS  = OFF_QREF_VEL + FLOAT_BLOCK
OFF_ACTUAL_VEL  = OFF_ACTUAL_POS + FLOAT_BLOCK
OFF_CMD         = OFF_ACTUAL_VEL + FLOAT_BLOCK
OFF_RESIDUAL    = OFF_CMD + CMD_BLOCK
OFF_COMBINED    = OFF_RESIDUAL + FLOAT_BLOCK
OFF_CTRL        = OFF_COMBINED + FLOAT_BLOCK
SHM_SIZE        = OFF_CTRL + FLOAT_BLOCK


def read_shm(buf):
    seq = struct.unpack_from("<I", buf, OFF_SEQ)[0]
    ts  = struct.unpack_from("<Q", buf, OFF_TS)[0]
    ctrl = np.frombuffer(buf, dtype=np.float32, count=NUM_JOINTS, offset=OFF_CTRL).copy()
    return seq, ts, ctrl


def main():
    parser = argparse.ArgumentParser(description="Log per-joint jitter (std) from CMG shared memory")
    parser.add_argument("--window", type=float, default=2.0, help="Rolling window in seconds (default 2)")
    parser.add_argument("--rate", type=float, default=200.0, help="Sampling rate in Hz (default 200)")
    parser.add_argument("--csv", type=str, default=None, help="Save raw samples to CSV file")
    parser.add_argument("--print-interval", type=float, default=2.0, help="Print interval in seconds")
    args = parser.parse_args()

    # Open shared memory via shm_open (no extra deps)
    fd = _rt.shm_open(SHM_NAME.encode(), os.O_RDONLY, 0)
    if fd < 0:
        print(f"[ERROR] Shared memory {SHM_NAME} not found. Is the controller running?")
        sys.exit(1)
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)

    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w")
        header = "timestamp_us," + ",".join(f"ctrl_{j}" for j in JOINT_NAMES)
        csv_file.write(header + "\n")

    dt = 1.0 / args.rate
    max_samples = int(args.window * args.rate)
    ctrl_buf = np.zeros((max_samples, NUM_JOINTS), dtype=np.float32)
    idx = 0
    count = 0
    last_seq = 0
    last_print = time.time()

    print(f"[jitter_logger] window={args.window}s  rate={args.rate}Hz  max_samples={max_samples}")
    print(f"[jitter_logger] Waiting for data on {SHM_NAME}...")

    try:
        while True:
            seq, ts, ctrl = read_shm(buf)
            if seq == last_seq:
                time.sleep(dt * 0.5)
                continue
            last_seq = seq

            ctrl_buf[idx % max_samples] = ctrl
            idx += 1
            count = min(count + 1, max_samples)

            if csv_file:
                line = f"{ts}," + ",".join(f"{v:.6f}" for v in ctrl)
                csv_file.write(line + "\n")

            now = time.time()
            if now - last_print >= args.print_interval and count >= 10:
                last_print = now
                valid = ctrl_buf[:count] if count < max_samples else ctrl_buf
                ctrl_std = np.std(valid, axis=0)

                print(f"\n{'='*56}")
                print(f"  Control Jitter (std) | samples={count} | window={args.window}s")
                print(f"{'='*56}")
                print(f"  {'Joint':<26s} {'ctrl_std (Nm)':>14s}")
                print(f"  {'-'*26} {'-'*14}")
                for i, name in enumerate(JOINT_NAMES):
                    print(f"  {name:<26s} {ctrl_std[i]:14.6f}")
                print(f"  {'-'*26} {'-'*14}")
                print(f"  {'MEAN':<26s} {np.mean(ctrl_std):14.6f}")
                print(f"  {'MAX':<26s} {np.max(ctrl_std):14.6f}")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[joint_jitter_logger] Stopped.")
    finally:
        buf.close()
        if csv_file:
            csv_file.close()
            print(f"[joint_jitter_logger] CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
