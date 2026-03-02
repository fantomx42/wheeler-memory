"""CLI: Interactive brick timeline viewer with matplotlib slider.

Usage:
    wheeler-scrub path/to/brick.npz
    wheeler-scrub --text "some text"  # auto-locate brick by hash
    wheeler-scrub --text "some text" --chunk code  # look in specific chunk
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from wheeler_memory.brick import MemoryBrick
from wheeler_memory.chunking import find_brick_across_chunks, get_chunk_dir
from wheeler_memory.hashing import text_to_hex
from wheeler_memory.storage import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Scrub through a Wheeler Memory brick timeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("path", nargs="?", help="Path to brick .npz file")
    group.add_argument("--text", help="Look up brick by original text")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)")
    parser.add_argument("--chunk", default=None, help="Chunk to search in (default: search all)")
    args = parser.parse_args()

    if args.text:
        data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
        hex_key = text_to_hex(args.text)

        if args.chunk:
            chunk_dir = get_chunk_dir(data_dir, args.chunk)
            brick_path = chunk_dir / "bricks" / f"{hex_key}.npz"
            if not brick_path.exists():
                print(f"No brick found for text: {args.text!r} in chunk '{args.chunk}'")
                print(f"  Expected: {brick_path}")
                return
        else:
            brick_path = find_brick_across_chunks(hex_key, data_dir)
            if brick_path is None:
                print(f"No brick found for text: {args.text!r}")
                print(f"  Searched all chunks in: {data_dir / 'chunks'}")
                return
    else:
        brick_path = Path(args.path)
        if not brick_path.exists():
            print(f"File not found: {brick_path}")
            return

    brick = MemoryBrick.load(brick_path)
    n_ticks = len(brick.evolution_history)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    im = ax.imshow(brick.evolution_history[0], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"Tick 0 / {n_ticks - 1}  |  State: {brick.state}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, "Tick", 0, n_ticks - 1, valinit=0, valstep=1)

    def update(val):
        tick = int(slider.val)
        im.set_data(brick.evolution_history[tick])
        ax.set_title(f"Tick {tick} / {n_ticks - 1}  |  State: {brick.state}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
