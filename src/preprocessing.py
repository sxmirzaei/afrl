
import argparse
import os
from typing import Optional, List
import numpy as np
import pandas as pd


def resample(df: pd.DataFrame, sampling_time: float):
    """
    This function takes the block average as a simple way to reduce noise
    """
    df = df.sort_values("timestamp")
    bins = np.arange(
        df['timestamp'].min(), 
        df['timestamp'].max() + sampling_time + 1e-9, 
        sampling_time
    )
    
    df["bin_index"] = np.digitize(df["timestamp"], bins, right=True)

    grouped = df.groupby("bin_index")
    mean_values = grouped[["tx", "ty", "tz"]].mean()

    unique_grouped_indices = mean_values.index.values

    valid_indices_mask = unique_grouped_indices > 0
    
    filtered_bin_indices = unique_grouped_indices[valid_indices_mask]
    filtered_mean_values = mean_values[valid_indices_mask]

    df_resampled = pd.DataFrame({
        # For a bin_index 'k', the block starts at bins[k-1]
        'timestamp': bins[filtered_bin_indices - 1],
        'tx': filtered_mean_values["tx"].values,
        'ty': filtered_mean_values["ty"].values,
        'tz': filtered_mean_values["tz"].values
    })
    
    return df_resampled

def pos_to_vel(df: pd.DataFrame):
    out = pd.DataFrame(columns=["timestamp", "vx", "vy", "vz"])
    dt = df["timestamp"].diff()

    out["timestamp"] = df["timestamp"]
    out["vx"] = df["tx"].diff() / dt
    out["vy"] = df["ty"].diff() / dt
    out["vz"] = df["tz"].diff() / dt

    return out.iloc[1:]

def vel_to_acc(df: pd.DataFrame):
    out = pd.DataFrame(columns=["timestamp", "ax", "ay", "az"])
    dt = df["timestamp"].diff()

    out["timestamp"] = df["timestamp"]
    out["ax"] = df["vx"].diff() / dt
    out["ay"] = df["vy"].diff() / dt
    out["az"] = df["vz"].diff() / dt

    return out.iloc[1:]


def walk_and_process(
    root: str,
    out_path_pos: str,
    out_path_vel: Optional[str],
    out_path_acc: Optional[str],
    sampling_time: float,
):
    for dirname in os.listdir(root):
        source_dir = os.path.join(root, dirname)
        if not os.path.isdir(source_dir):
            continue

        # Position is always resampled
        os.makedirs(os.path.join(out_path_pos, dirname), exist_ok=True)
        if out_path_vel:
            os.makedirs(os.path.join(out_path_vel, dirname), exist_ok=True)
        if out_path_acc:
            os.makedirs(os.path.join(out_path_acc, dirname), exist_ok=True)

        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            if not os.path.isfile(source_file):
                continue

            df = pd.read_csv(source_file)

            pos = resample(df, sampling_time)
            pos.to_csv(os.path.join(out_path_pos, dirname, filename), index=False)

            need_velocity = bool(out_path_vel) or bool(out_path_acc)
            vel = None
            if need_velocity:
                vel = pos_to_vel(pos)
                if out_path_vel:
                    vel.to_csv(os.path.join(out_path_vel, dirname, filename), index=False)

            if out_path_acc and vel is not None:
                acc = vel_to_acc(vel)
                acc.to_csv(os.path.join(out_path_acc, dirname, filename), index=False)


def scale_by(df: pd.DataFrame, coords: List[str], max):
    df[coords] = df[coords] / max

def max_mag(df: pd.DataFrame, coords: List[str]):
    coord_data = df[coords]

    sum_of_squares = (coord_data**2).sum(axis=1)
    magnitudes = np.sqrt(sum_of_squares)

    return magnitudes.max()


def walk_and_normalize(root: str, out: str, coords: List[str]):
    os.makedirs(out, exist_ok=True)

    max = 0
    for dirname in os.listdir(root):
        for filename in os.listdir(os.path.join(root, dirname)):
            curr = max_mag(
                pd.read_csv(
                    os.path.join(root, dirname, filename), 
                    usecols=["timestamp"] + coords
                    ), 
                coords
            )

            max = curr if curr > max else max
    
    for dirname in os.listdir(root):
        src_dir = os.path.join(root, dirname)
        if not os.path.isdir(src_dir):
            continue

        os.makedirs(os.path.join(out, dirname), exist_ok=True)
        for filename in os.listdir(src_dir):
            df = pd.read_csv(
                os.path.join(src_dir, filename), 
                usecols=["timestamp"] + coords
            )
            scale_by(df, coords, max)
            df.to_csv(os.path.join(out, dirname, filename), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Derive velocity and acceleration data. Normalize said data."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Derives vel and acc, then normalizes vel and acc."
        "NOTE: This does not specify which datasets to include, just what actions to perform on the datasets."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/clean",
        help="Input root directory containing raw position CSVs (default: data/clean).",
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="data",
        help="Output root directory where processed data will be saved (default: data).",
    )
    
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=0.1,
        help="Resamples position data. Defaults to 0.1 sec or 10hz."
    )

    parser.add_argument(
        "-v",
        "--velocity",
        action="store_true",
        help="Derives velocity from position."
    )

    parser.add_argument(
        "-a",
        "--acceleration",
        action="store_true",
        help="Derives acceleration from velocity. Fails if velocity does not exist."
    )

    parser.add_argument(
        "-np",
        "--norm-position",
        action="store_true",
        help="Normalizes position data."
    )

    parser.add_argument(
        "-nv",
        "--norm-velocity",
        action="store_true",
        help="Normalizes velocity data."
    )

    parser.add_argument(
        "-na",
        "--norm-acceleration",
        action="store_true",
        help="Normalizes acceleration data."
    )

    args = parser.parse_args()

    data_root = args.data_root
    pos_path = os.path.join(args.out_root, "position", "raw")
    vel_path = os.path.join(args.out_root, "velocity", "raw")
    acc_path = os.path.join(args.out_root, "acceleration", "raw")

    os.makedirs(pos_path, exist_ok=True)
    if args.all or args.velocity:
        os.makedirs(vel_path, exist_ok=True)
    if args.all or args.acceleration:
        os.makedirs(acc_path, exist_ok=True)

    walk_and_process(
        root=data_root,
        out_path_pos=pos_path,
        out_path_vel=vel_path if args.all or args.velocity or args.acceleration else None,
        out_path_acc=acc_path if args.all or args.acceleration else None,
        sampling_time=args.time,
    )
    print("Done resampling and deriving.")

    if args.all or args.norm_position:
        pos_norm_path = "data/position/max_norm"
        walk_and_normalize(
            root=pos_path,
            out=pos_norm_path,
            coords=["tx", "ty", "tz"]
        )
        print("Done normalizing position.")

    if args.all or args.norm_velocity:
        vel_norm_path = "data/velocity/max_norm"
        walk_and_normalize(
            root=vel_path,
            out=vel_norm_path,
            coords=["vx", "vy", "vz"]
        )
        print("Done normalizing velocity.")

    if args.all or args.norm_acceleration:
        acc_norm_path = "data/acceleration/max_norm"
        walk_and_normalize(
            root=acc_path,
            out=acc_norm_path,
            coords=["ax", "ay", "az"]
        )
        print("Done normalizing acceleration.")
    
    print("Finished.")
    
if __name__ == "__main__":
    main()