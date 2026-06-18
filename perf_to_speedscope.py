"""Convert `perf script` stack samples to a Speedscope sampled profile JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EVENT_RE = re.compile(r"\s+(\d+(?:\.\d+)?):\s+")


def clean_frame(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    if text.startswith("0x"):
        parts = text.split(None, 1)
        return parts[1] if len(parts) > 1 else parts[0]
    return text


def parse_perf_script(path: Path) -> tuple[list[float], list[list[str]]]:
    times: list[float] = []
    stacks: list[list[str]] = []
    current_time: float | None = None
    current_stack: list[str] = []

    def flush() -> None:
        nonlocal current_time, current_stack
        if current_time is not None and current_stack:
            times.append(current_time)
            stacks.append(list(reversed(current_stack)))
        current_time = None
        current_stack = []

    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip():
            flush()
            continue
        match = EVENT_RE.search(raw)
        if match and not raw.startswith("\t") and not raw.startswith(" " * 8):
            flush()
            current_time = float(match.group(1))
            continue
        frame = clean_frame(raw)
        if frame:
            current_stack.append(frame)
    flush()
    return times, stacks


def build_speedscope(times: list[float], stacks: list[list[str]], name: str) -> dict:
    frame_ids: dict[str, int] = {}
    frames: list[dict[str, str]] = []
    samples: list[list[int]] = []

    for stack in stacks:
        sample: list[int] = []
        for frame in stack:
            if frame not in frame_ids:
                frame_ids[frame] = len(frames)
                frames.append({"name": frame})
            sample.append(frame_ids[frame])
        samples.append(sample)

    if times:
        start = times[0]
        end = times[-1]
        weights = [
            max(1.0e-9, (times[i + 1] - times[i]) if i + 1 < len(times) else (times[i] - times[i - 1] if i else 1.0e-3))
            for i in range(len(times))
        ]
    else:
        start = 0.0
        end = 0.0
        weights = []

    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {"frames": frames},
        "profiles": [
            {
                "type": "sampled",
                "name": name,
                "unit": "seconds",
                "startValue": start,
                "endValue": end,
                "samples": samples,
                "weights": weights,
            }
        ],
        "activeProfileIndex": 0,
        "exporter": "perf_to_speedscope.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="perf script text output")
    parser.add_argument("output", type=Path, help="Speedscope JSON output")
    parser.add_argument("--name", default="perf profile")
    args = parser.parse_args()

    times, stacks = parse_perf_script(args.input)
    if not stacks:
        raise SystemExit(f"no stacks found in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_speedscope(times, stacks, args.name)))
    print(f"wrote {args.output} with {len(stacks)} samples")


if __name__ == "__main__":
    main()
