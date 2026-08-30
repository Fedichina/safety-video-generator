"""
video_generator.py
-------------------
Core text -> narrated safety video pipeline, reused by the Flask web app.
"""

import asyncio
import os
import subprocess
import tempfile
import requests
import edge_tts
from moviepy import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    ColorClip, concatenate_videoclips
)

VIDEO_SIZE = (1080, 1920)
FONT_SIZE = 58


def fetch_stock_clip(query: str, out_path: str, api_key: str) -> bool:
    if not api_key:
        return False
    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 1, "orientation": "portrait"},
            timeout=15,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if not videos:
            return False
        files = sorted(videos[0]["video_files"], key=lambda f: f.get("width", 0))
        candidates = [f for f in files if 640 <= f.get("width", 0) <= 1920] or files
        video_url = candidates[-1]["link"]

        with requests.get(video_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception:
        return False


async def _make_narration(text: str, out_path: str, voice: str):
    await edge_tts.Communicate(text, voice).save(out_path)


async def _make_narration_with_timeout(text: str, out_path: str, voice: str, timeout: float = 30.0):
    try:
        await asyncio.wait_for(_make_narration(text, out_path, voice), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Narration generation timed out after {timeout}s — "
            "the text-to-speech service may be slow or unreachable right now."
        )


def _mp3_to_wav(mp3_path: str, wav_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "44100", "-ac", "2", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_scene(text: str, index: int, tmpdir: str, api_key: str, voice: str):
    mp3_path = os.path.join(tmpdir, f"audio_{index}.mp3")
    wav_path = os.path.join(tmpdir, f"audio_{index}.wav")
    asyncio.run(_make_narration(text, mp3_path, voice))
    _mp3_to_wav(mp3_path, wav_path)

    audio_clip = AudioFileClip(wav_path)
    duration = audio_clip.duration

    video_path = os.path.join(tmpdir, f"stock_{index}.mp4")
    keyword = text.split(",")[0][:40]
    got_stock = fetch_stock_clip(keyword or "workplace safety", video_path, api_key)

    if got_stock:
        bg = VideoFileClip(video_path).resized(height=VIDEO_SIZE[1])
        bg = bg.cropped(x_center=bg.w / 2, width=VIDEO_SIZE[0])
        if bg.duration < duration:
            n_loops = int(duration // bg.duration) + 1
            bg = concatenate_videoclips([bg] * n_loops)
        bg = bg.subclipped(0, duration)
    else:
        bg = ColorClip(size=VIDEO_SIZE, color=(20, 30, 40), duration=duration)

    caption = (
        TextClip(
            text=text,
            font_size=FONT_SIZE,
            color="white",
            size=(VIDEO_SIZE[0] - 120, None),
            method="caption",
            stroke_color="black",
            stroke_width=2,
        )
        .with_duration(duration)
        .with_position(("center", "bottom"))
    )

    scene = CompositeVideoClip([bg, caption], size=VIDEO_SIZE).with_audio(audio_clip)
    return scene.with_duration(duration)


def generate_video(script_text: str, output_path: str, api_key: str = "",
                    voice: str = "en-US-GuyNeural", progress_cb=None):
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    lines = [l.strip() for l in script_text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("No scenes found — please enter at least one line of text.")

    with tempfile.TemporaryDirectory() as tmpdir:
        clips = []
        for i, line in enumerate(lines):
            report(f"Generating scene {i + 1} of {len(lines)}: \"{line[:60]}\"")
            clips.append(build_scene(line, i, tmpdir, api_key, voice))

        report("Assembling final video...")
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac",
                               preset="ultrafast", threads=4, logger=None)
        report("Done!")

    return output_path
