"""Generate the synthetic 90-second demo clip (no real people, places, or plates).

Scene timeline (seconds):
  10-22  blue sedan crosses left-to-right; white rear plate with dark glyph bars
  13.67-14.13  focus hunt (frames 410-424): stacked Gaussian blur peaking at frame 417,
         the reconstruction target; frames 367-409 and 425-477 keep a sharp, fully
         visible plate (detector-backed donors within +/-2 s)
  40-52  red van crosses right-to-left; its plate is covered by a gray occluding bar
  60-75  static green sign with glyph bars, upper right
  78-84  scene darkens (lighting change)

Usage: uv run python scripts/generate_demo_video.py fixtures/demo/source/demo-plate-90s.mp4
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

W, H, FPS, DURATION = 1280, 720, 30, 90
# (sigma, first_frame, last_frame); stacked Gaussians add in quadrature (peak ~3.35 px at 417).
FOCUS_HUNT = ((1.2, 410, 424), (1.2, 413, 421), (1.6, 415, 419), (2.4, 417, 417))


def box(x: str, y: str, w: int, h: int, color: str, enable: str) -> str:
    return f"drawbox=x='{x}':y='{y}':w={w}:h={h}:color={color}:t=fill:enable='{enable}'"


def sprite(w: int, h: int, boxes: list[str]) -> str:
    opaque = [b + ":replace=1" for b in boxes]
    return f"color=c=black@0.0:s={w}x{h}:r={FPS}:d={DURATION},format=rgba," + ",".join(opaque)


def filters() -> str:
    background = [
        box("0", "420", W, 220, "0x3a3a3a", "1"),
        *[box(str(x), "525", 80, 10, "0xd8d8d8", "1") for x in range(40, W, 200)],
        box("980", "120", 200, 90, "0x1d7a3a", "between(t,60,75)"),
        box("1076", "210", 8, 200, "0x888888", "between(t,60,75)"),
        *[box(str(1000 + i * 34), "150", 18, 30, "0xf5f5f5", "between(t,60,75)") for i in range(5)],
    ]
    sedan = sprite(
        280,
        160,
        [
            box("0", "40", 280, 100, "0x1f4fbf", "1"),
            box("70", "0", 150, 50, "0x173a8c", "1"),
            box("10", "80", 76, 26, "0xf2f2f2", "1"),
            *[box(str(16 + i * 13), "85", 7, 16, "0x101010", "1") for i in range(5)],
            box("30", "128", 50, 30, "0x111111", "1"),
            box("200", "128", 50, 30, "0x111111", "1"),
        ],
    )
    van = sprite(
        320,
        180,
        [
            box("0", "0", 320, 150, "0xa82020", "1"),
            box("234", "90", 76, 26, "0xf2f2f2", "1"),
            box("220", "82", 100, 44, "0x777777", "1"),
            box("40", "140", 56, 32, "0x111111", "1"),
            box("230", "140", 56, 32, "0x111111", "1"),
        ],
    )
    return ";".join(
        [
            f"[0:v]{','.join(background)}[bg]",
            f"{sedan}[sedan]",
            f"{van}[van]",
            "[bg][sedan]overlay=x='-300+(t-10)*130':y=390:eval=frame:enable='between(t,10,22)'[a]",
            "[a][van]overlay=x='1300-(t-40)*135':y=380:eval=frame:enable='between(t,40,52)'[b]",
            "[b]eq=brightness=-0.35:enable='between(t,78,84)',"
            + ",".join(
                f"gblur=sigma={sigma}:enable='between(n,{first},{last})'"
                for sigma, first, last in FOCUS_HUNT
            )
            + ",format=yuv420p[out]",
        ]
    )


def main(out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x6f8f6f:s={W}x{H}:r={FPS}:d={DURATION}",
        "-filter_complex",
        filters(),
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "60",
        "-threads",
        "1",
        "-x264-params",
        "threads=1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        "-an",
        out,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/demo/source/demo-plate-90s.mp4")
