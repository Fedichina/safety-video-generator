"""
app.py — Safety Video Generator web app.

Run locally:
    export PEXELS_API_KEY="your_key_here"   # optional, enables real stock footage
    python3 app.py
Then open http://localhost:5000
"""

import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file, url_for

from video_generator import generate_video

app = Flask(__name__)

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

JOBS = {}

VOICES = {
    "en-US-GuyNeural": "Guy (US, male)",
    "en-US-JennyNeural": "Jenny (US, female)",
    "en-GB-RyanNeural": "Ryan (UK, male)",
    "en-GB-SoniaNeural": "Sonia (UK, female)",
    "en-AU-WilliamNeural": "William (AU, male)",
}


@app.route("/")
def index():
    return render_template("index.html", voices=VOICES)


@app.route("/generate", methods=["POST"])
def generate():
    script_text = request.form.get("script", "").strip()
    voice = request.form.get("voice", "en-US-GuyNeural")
    api_key = request.form.get("api_key", "").strip() or os.environ.get("PEXELS_API_KEY", "")

    if not script_text:
        return jsonify({"error": "Please enter a script."}), 400

    job_id = uuid.uuid4().hex[:12]
    output_path = os.path.join(GENERATED_DIR, f"{job_id}.mp4")
    JOBS[job_id] = {"status": "running", "message": "Starting...", "file": None, "error": None}

    def worker():
        try:
            def progress(msg):
                JOBS[job_id]["message"] = msg

            generate_video(script_text, output_path, api_key=api_key, voice=voice,
                            progress_cb=progress)
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["file"] = f"{job_id}.mp4"
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    resp = {"status": job["status"], "message": job["message"]}
    if job["status"] == "done":
        resp["video_url"] = url_for("get_video", job_id=job_id)
    if job["status"] == "error":
        resp["error"] = job["error"]
    return jsonify(resp)


@app.route("/video/<job_id>")
def get_video(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return "Not ready", 404
    return send_file(os.path.join(GENERATED_DIR, job["file"]), mimetype="video/mp4")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
