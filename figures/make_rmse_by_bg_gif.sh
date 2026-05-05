#!/usr/bin/env bash
# Generate one rmse_vs_bg_h{H}.png per 5-min horizon (5..60) and fuse them into a gif.
#
# Usage:
#   bash figures/make_rmse_by_bg_gif.sh [INPUT_PARQUET] [OUT_GIF] [FRAME_DELAY_CS] [FRAME_DIR]
#
#   INPUT_PARQUET   default: combined_results_with_aux.parquet
#   OUT_GIF         default: rmse_vs_bg.gif
#   FRAME_DELAY_CS  default: 60   (centiseconds per frame; ImageMagick `-delay`)
#   FRAME_DIR       default: figures/rmse_vs_bg_frames
#
# Requires either ImageMagick (`magick` or `convert`) or `ffmpeg` on PATH.

set -euo pipefail

INPUT="${1:-combined_results_with_aux.parquet}"
OUT_GIF="${2:-rmse_vs_bg.gif}"
DELAY_CS="${3:-60}"
FRAME_DIR="${4:-figures/rmse_vs_bg_frames}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_rmse_by_bg.py"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "Error: neither python nor python3 found on PATH." >&2
        exit 1
    fi
fi

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input parquet not found: $INPUT" >&2
    exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "Error: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$FRAME_DIR"

frames=()
for H in 5 10 15 20 25 30 35 40 45 50 55 60; do
    OUT_PNG="$FRAME_DIR/rmse_vs_bg_h${H}.png"
    echo "[$(printf '%2d' "$H") min] -> $OUT_PNG"
    "$PYTHON" "$PLOT_SCRIPT" \
        --input "$INPUT" \
        --combined \
        --output "$OUT_PNG" \
        --horizon "$H" \
        --ablation exclude
    frames+=("$OUT_PNG")
done

echo "Fusing ${#frames[@]} frames into $OUT_GIF..."

if command -v magick >/dev/null 2>&1; then
    magick -delay "$DELAY_CS" -loop 0 "${frames[@]}" "$OUT_GIF"
elif command -v convert >/dev/null 2>&1; then
    convert -delay "$DELAY_CS" -loop 0 "${frames[@]}" "$OUT_GIF"
elif command -v ffmpeg >/dev/null 2>&1; then
    # ffmpeg uses fps; convert centiseconds-per-frame to frames-per-second.
    fps=$(awk -v d="$DELAY_CS" 'BEGIN { printf "%.4f", 100.0 / d }')
    pattern="$FRAME_DIR/rmse_vs_bg_h%d.png"
    # ffmpeg's %d glob matches the integer horizon directly.
    ffmpeg -y -framerate "$fps" -i "$pattern" \
        -vf "split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse" \
        -loop 0 "$OUT_GIF"
else
    echo "Error: need ImageMagick (magick/convert) or ffmpeg installed to build the gif." >&2
    echo "Frames are saved in $FRAME_DIR." >&2
    exit 1
fi

echo "Done: $OUT_GIF"
