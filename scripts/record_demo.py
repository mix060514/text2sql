"""Record the README demo GIF: ask one question on the Chat page and click through the flow.

Needs the live model to be idle (one request at a time) plus Playwright and ffmpeg:

    uvx --with playwright==1.55.0 python scripts/record_demo.py
    uvx --with playwright==1.55.0 python scripts/record_demo.py --dry-run   # no model call

The model wait is sped up so the GIF stays short; the answer, chart and flow are real.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
QUESTION = "2024 年各地區的銷售額分別是多少？"


def wait_http(url: str, seconds: float = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


def record(port: int, video_dir: Path, dry_run: bool, answer_seconds: float, chromium: str | None) -> dict:
    marks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium)
        context = browser.new_context(viewport={"width": 1280, "height": 860}, color_scheme="light",
                                      record_video_dir=str(video_dir),
                                      record_video_size={"width": 1280, "height": 860})
        page = context.new_page()
        start = time.monotonic()
        mark = lambda name: marks.__setitem__(name, round(time.monotonic() - start, 2))
        page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        page.get_by_role("link", name="對話查詢").click()
        box = page.locator("[data-testid='stChatInputTextArea']")
        try:
            box.wait_for(timeout=30_000)
        except Exception:
            page.screenshot(path=str(video_dir / "failed.png"))
            raise RuntimeError(f"chat input not found; screenshot in {video_dir}") from None
        page.wait_for_timeout(1200)
        box.click()
        box.press_sequentially(QUESTION, delay=60)
        page.wait_for_timeout(600)
        if dry_run:
            page.wait_for_timeout(1500)
            mark("end")
        else:
            box.press("Enter")
            mark("submitted")
            page.wait_for_timeout(4000)  # the live flow starts drawing; keep this part at real speed
            mark("speedup_from")
            page.locator("[data-testid='stPlotlyChart']").first.wait_for(timeout=answer_seconds * 1000)
            mark("answered")
            page.wait_for_timeout(2500)
            frame = page.frame_locator("iframe").last
            frame.locator("#flow").scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            for step in ("agent_review", "step_run_sql", "agent_sql"):
                frame.locator("#flow g", has_text=step).first.click()
                page.wait_for_timeout(2600)
            mark("end")
        video = page.video
        context.close()
        browser.close()
        marks["video"] = video.path()
    return marks


def to_gif(video: Path, marks: dict, out: Path, speedup_to: float = 4.0) -> None:
    """Keep real speed except the model wait, which is squeezed to `speedup_to` seconds."""
    if "speedup_from" in marks:
        a, b = marks["speedup_from"], max(marks["answered"] - 0.5, marks["speedup_from"] + 1)
        factor = (b - a) / speedup_to
        graph = (f"[0:v]trim=0:{a},setpts=PTS-STARTPTS[v0];"
                 f"[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)/{factor:.3f}[v1];"
                 f"[0:v]trim={b},setpts=PTS-STARTPTS[v2];"
                 "[v0][v1][v2]concat=n=3:v=1[v];")
    else:
        graph = "[0:v]null[v];"
    graph += ("[v]fps=10,scale=960:-1:flags=lanczos,split[s0][s1];"
              "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
              "[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-filter_complex", graph, str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT / "assets" / "demo.gif")
    parser.add_argument("--port", type=int, default=8517)
    parser.add_argument("--dry-run", action="store_true", help="type the question but do not send it")
    parser.add_argument("--answer-seconds", type=float, default=400)
    parser.add_argument("--chromium", help="Chromium executable, if Playwright's own build is not installed")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="demo-"))
    server = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "web_ui/app.py", "--server.headless=true",
         f"--server.port={args.port}", "--server.address=127.0.0.1", "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal"],
        cwd=PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://127.0.0.1:{args.port}/_stcore/health")
        marks = record(args.port, work, args.dry_run, args.answer_seconds, args.chromium)
    finally:
        server.terminate()
        server.wait(timeout=20)
    out = args.out.with_name(args.out.stem + "-dry.gif") if args.dry_run else args.out
    to_gif(Path(marks["video"]), marks, out)
    print(json.dumps({**marks, "gif": str(out), "bytes": out.stat().st_size}, ensure_ascii=False))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
