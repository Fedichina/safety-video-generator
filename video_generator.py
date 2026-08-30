"""
video_generator.py
-------------------
Core text -> narrated safety video pipeline, reused by the Flask web app.

Free/cheap tools used:
  - edge-tts        : free neural narration
  - Pexels API      : free stock video search (needs a free API key)
  - Wan-AI/Wan2.1   : free AI video generation via Hugging Face Spaces (experimental)
  - moviepy         : free, open-source video assembly

Memory strategy: each scene is rendered to its own small mp4 file and
released from memory immediately, rather than holding every scene's
clip objects in RAM until the end. This matters a lot on constrained
hosts (e.g. Render's free 512MB tier).
"""

import asyncio
import gc
import os
import subprocess
import tempfile
import requests
import edge_tts
from moviepy import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    ColorClip, concatenate_videoclips
)

VIDEO_SIZE = (1080, 1920)  # vertical 9:16, for Shorts/Reels/TikTok
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
        candidates = [f for f in files if 480 <= f.get("width", 0) <= 720] or files
        video_url = candidates[0]["link"]

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


def generate_ai_clip(prompt: str, out_path: str, hf_token: str = "", timeout: float = 180.0) -> bool:
    """
    Generate a short video clip from a text prompt using the free
    Wan-AI/Wan2.1 model hosted on Hugging Face Spaces (their shared
    free GPU quota — "ZeroGPU"). Experimental: can be slow, rate
    limited, and lower quality than paid tools, but free.
    """
    try:
        from gradio_client import Client
        client = Client("Wan-AI/Wan2.1", hf_token=hf_token or None)

        result = client.predict(
            prompt,
            None,
            "832*480",
            False,
            -1,
            api_name="/generate_video",
        )

        if isinstance(result, dict) and "video" in result:
            src_path = result["video"]
        elif isinstance(result, (list, tuple)):
            src_path = result[0]
        else:
            src_path = result

        if not src_path or not os.path.exists(src_path):
            return False

        subprocess.run(["cp", src_path, out_path], check=True)
        return True

    except Exception:
        return False


def _mp3_to_wav(mp3_path: str, wav_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "44100", "-ac", "2", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_scene_to_file(text: str, index: int, tmpdir: str, api_key: str, voice: str,
                          search_query: str = None, hf_token: str = "") -> str:
    """
    search_query: optional explicit background instruction.
    - Starts with "ai:" -> free AI video generation prompt.
    - Otherwise -> Pexels search phrase.
    - Not given -> auto-extracted keyword from narration text.
    """
    mp3_path = os.path.join(tmpdir, f"audio_{index}.mp3")
    wav_path = os.path.join(tmpdir, f"audio_{index}.wav")
    asyncio.run(_make_narration_with_timeout(text, mp3_path, voice))
    _mp3_to_wav(mp3_path, wav_path)

    audio_clip = AudioFileClip(wav_path)
    duration = audio_clip.duration

    video_path = os.path.join(tmpdir, f"stock_{index}.mp4")
    bg = None

    use_ai = bool(search_query) and search_query.strip().lower().startswith("ai:")

    if use_ai:
        ai_prompt = search_query.strip()[3:].strip()
        got_ai = generate_ai_clip(ai_prompt, video_path, hf_token=hf_token)
        if got_ai:
            try:
                bg = VideoFileClip(video_path).resized(height=VIDEO_SIZE[1])
                bg = bg.cropped(x_center=bg.w / 2, width=VIDEO_SIZE[0])
                if bg.duration < duration:
                    n_loops = int(duration // bg.duration) + 1
                    bg = concatenate_videoclips([bg] * n_loops)
                bg = bg.subclipped(0, duration)
            except Exception:
                bg = None
    else:
        keyword = search_query.strip() if search_query else text.split(",")[0][:40]
        got_stock = fetch_stock_clip(keyword or "workplace safety", video_path, api_key)
        if got_stock:
            try:
                bg = VideoFileClip(video_path).resized(height=VIDEO_SIZE[1])
                bg = bg.cropped(x_center=bg.w / 2, width=VIDEO_SIZE[0])
                if bg.duration < duration:
                    n_loops = int(duration // bg.duration) + 1
                    bg = concatenate_videoclips([bg] * n_loops)
                bg = bg.subclipped(0, duration)
            except Exception:
                bg = None

    if bg is None:
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
    scene = scene.with_duration(duration)

    scene_output_path = os.path.join(tmpdir, f"scene_{index}.mp4")
    scene.write_videofile(
        scene_output_path, fps=24, codec="libx264", audio_codec="aac",
        preset="ultrafast", threads=1, logger=None,
    )

    try:
        scene.close()
    except Exception:
        pass
    try:
        audio_clip.close()
    except Exception:
        pass
    try:
        bg.close()
    except Exception:
        pass
    del scene, audio_clip, bg, caption
    gc.collect()

    return scene_output_path


def generate_video(script_text: str, output_path: str, api_key: str = "",
                    voice: str = "en-US-GuyNeural", hf_token: str = "", progress_cb=None):
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    lines = [l.strip() for l in script_text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("No scenes found — please enter at least one line of text.")

    with tempfile.TemporaryDirectory() as tmpdir:
        scene_files = []
        for i, raw_line in enumerate(lines):
            if "|" in raw_line:
                caption_text, search_query = raw_line.split("|", 1)
                caption_text = caption_text.strip()
                search_query = search_query.strip()
            else:
                caption_text = raw_line
                search_query = None

            report(f"Generating scene {i + 1} of {len(lines)}: \"{caption_text[:60]}\"")
            scene_path = render_scene_to_file(caption_text, i, tmpdir, api_key, voice,
                                               search_query=search_query, hf_token=hf_token)
            scene_files.append(scene_path)

        report("Assembling final video...")
        concat_list_path = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for sp in scene_files:
                f.write(f"file '{sp}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        report("Done!")

    return output_path
