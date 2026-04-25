#!/usr/bin/env python3
"""
Quick visualization helpers for FastView tracking JSON files.

The first plot is a faceted body-center track view: one subplot per fly ID.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fly_sort_key(fly_id: str):
    return int(fly_id.split("_")[-1]) if "_" in fly_id and fly_id.split("_")[-1].isdigit() else fly_id


def read_tracking_json(json_path: str | Path) -> pd.DataFrame:
    """Read semicolon-separated tracking JSON into a long table."""
    rows = []
    path = Path(json_path)
    for part in path.read_text(encoding="utf-8", errors="replace").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            frame_obj = json.loads(part)
        except json.JSONDecodeError:
            continue
        for frame, flies in frame_obj.items():
            frame_i = int(frame)
            for fly_id, data in flies.items():
                body = data.get("body")
                head = data.get("head")
                if not body or len(body) < 2:
                    continue
                row = {
                    "frame": frame_i,
                    "fly_id": fly_id,
                    "body_x": float(body[0]),
                    "body_y": float(body[1]),
                    "body_w": float(body[2]) if len(body) > 2 else None,
                    "body_h": float(body[3]) if len(body) > 3 else None,
                }
                if head and len(head) >= 2:
                    row.update({
                        "head_x": float(head[0]),
                        "head_y": float(head[1]),
                    })
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["frame", "fly_id", "body_x", "body_y", "body_w", "body_h", "head_x", "head_y"])
    return pd.DataFrame(rows).sort_values(["fly_id", "frame"]).reset_index(drop=True)


def add_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-frame normalized body-center speed for each fly."""
    out = df.sort_values(["fly_id", "frame"]).copy()
    grp = out.groupby("fly_id", sort=False)
    out["prev_frame"] = grp["frame"].shift(1)
    out["dx"] = out["body_x"] - grp["body_x"].shift(1)
    out["dy"] = out["body_y"] - grp["body_y"].shift(1)
    out["dframe"] = out["frame"] - out["prev_frame"]
    out["speed"] = ((out["dx"] ** 2 + out["dy"] ** 2) ** 0.5) / out["dframe"].where(out["dframe"] > 0)
    return out


def plot_tracks_by_fly(
    json_path: str | Path,
    output: str | Path | None = None,
    cols: int = 4,
    point_size: float = 5.0,
    line_width: float = 0.8,
) -> Path:
    """Plot body-center tracks faceted by fly ID and save the figure."""
    df = read_tracking_json(json_path)
    if df.empty:
        raise ValueError(f"No track points found in {json_path}")

    fly_ids = sorted(df["fly_id"].unique(), key=fly_sort_key)
    cols = max(1, min(cols, len(fly_ids)))
    rows = math.ceil(len(fly_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), squeeze=False)
    frame_min = df["frame"].min()
    frame_max = df["frame"].max()

    for ax, fly_id in zip(axes.ravel(), fly_ids):
        sub = df[df["fly_id"] == fly_id]
        ax.plot(sub["body_x"], sub["body_y"], linewidth=line_width, color="#4c78a8", alpha=0.8)
        sc = ax.scatter(
            sub["body_x"],
            sub["body_y"],
            c=sub["frame"],
            cmap="viridis",
            s=point_size,
            alpha=0.9,
        )
        ax.set_title(f"{fly_id} ({len(sub)} pts)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes.ravel()[len(fly_ids):]:
        ax.axis("off")

    fig.suptitle(f"Fly Tracks: {Path(json_path).name} | frames {frame_min}-{frame_max}", fontsize=12)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Frame")
    if output is None:
        output = Path(json_path).with_suffix(".tracks_by_fly.png")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_speed_by_fly(
    json_path: str | Path,
    output: str | Path | None = None,
    cols: int = 4,
    line_width: float = 0.9,
    smooth_window: int = 30,
) -> Path:
    """Plot per-fly moving speed over frame number."""
    df = add_speed(read_tracking_json(json_path))
    df = df[df["speed"].notna()].copy()
    if df.empty:
        raise ValueError(f"No speed values could be calculated from {json_path}")
    smooth_window = max(1, int(smooth_window))
    df["speed_smooth"] = (
        df.groupby("fly_id", sort=False)["speed"]
        .transform(lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean())
    )

    fly_ids = sorted(df["fly_id"].unique(), key=fly_sort_key)
    cols = max(1, min(cols, len(fly_ids)))
    rows = math.ceil(len(fly_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.6), squeeze=False, sharex=True)
    y_max = df["speed_smooth"].quantile(0.99)
    if not pd.notna(y_max) or y_max <= 0:
        y_max = df["speed_smooth"].max()

    for ax, fly_id in zip(axes.ravel(), fly_ids):
        sub = df[df["fly_id"] == fly_id]
        ax.plot(sub["frame"], sub["speed_smooth"], linewidth=line_width, color="#f58518")
        ax.set_title(f"{fly_id}", fontsize=10)
        ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 1)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlabel("frame")
        ax.set_ylabel("speed")

    for ax in axes.ravel()[len(fly_ids):]:
        ax.axis("off")

    fig.suptitle(f"Fly Moving Speed: {Path(json_path).name} | rolling window {smooth_window}", fontsize=12)
    if output is None:
        output = Path(json_path).with_suffix(".speed_by_fly.png")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_total_speed(
    json_path: str | Path,
    output: str | Path | None = None,
    line_width: float = 1.2,
    smooth_window: int = 30,
) -> Path:
    """Plot total moving speed summed across all flies for each frame."""
    df = add_speed(read_tracking_json(json_path))
    df = df[df["speed"].notna()].copy()
    if df.empty:
        raise ValueError(f"No speed values could be calculated from {json_path}")
    smooth_window = max(1, int(smooth_window))
    total = (
        df.groupby("frame", as_index=False)["speed"]
        .sum()
        .sort_values("frame")
        .reset_index(drop=True)
    )
    total["speed_smooth"] = total["speed"].rolling(smooth_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total["frame"], total["speed_smooth"], linewidth=line_width, color="#54a24b")
    ax.set_title(f"Total Moving Speed: {Path(json_path).name} | rolling window {smooth_window}")
    ax.set_xlabel("frame")
    ax.set_ylabel("sum speed")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    if output is None:
        output = Path(json_path).with_suffix(".total_speed.png")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def default_plot_paths(json_path: str | Path, output_dir: str | Path | None) -> dict[str, Path]:
    stem = Path(json_path).name
    out_dir = Path(output_dir) if output_dir else Path(json_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "tracks": out_dir / f"{stem}.tracks_by_fly.png",
        "speed": out_dir / f"{stem}.speed_by_fly.png",
        "total_speed": out_dir / f"{stem}.total_speed.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastView tracking JSON visualization.")
    parser.add_argument("-j", "--json", required=True, help="Tracking JSON file from Post_track.py.")
    parser.add_argument("-o", "--output", default="", help="Output directory. Default: same directory as JSON.")
    parser.add_argument(
        "--plot",
        choices=["tracks", "speed", "total-speed", "both", "all"],
        default="all",
        help="Which plot to generate. 'all' saves tracks, per-fly speed, and total speed.",
    )
    parser.add_argument("--cols", type=int, default=4, help="Number of facet columns.")
    parser.add_argument("--point-size", type=float, default=5.0, help="Track point size.")
    parser.add_argument("--line-width", type=float, default=0.8, help="Track line width.")
    parser.add_argument("--speed-window", type=int, default=30, help="Rolling window size for smoothing speed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = default_plot_paths(args.json, args.output or None)
    if args.plot in {"tracks", "both", "all"}:
        out = plot_tracks_by_fly(
            args.json,
            output=outputs["tracks"],
            cols=args.cols,
            point_size=args.point_size,
            line_width=args.line_width,
        )
        print(f"Saved track facet plot: {out}")
    if args.plot in {"speed", "both", "all"}:
        speed_out = plot_speed_by_fly(
            args.json,
            output=outputs["speed"],
            cols=args.cols,
            line_width=args.line_width,
            smooth_window=args.speed_window,
        )
        print(f"Saved speed facet plot: {speed_out}")
    if args.plot in {"total-speed", "all"}:
        total_out = plot_total_speed(
            args.json,
            output=outputs["total_speed"],
            line_width=args.line_width,
            smooth_window=args.speed_window,
        )
        print(f"Saved total speed plot: {total_out}")


if __name__ == "__main__":
    main()
