import asyncio, json, os, subprocess, time, uuid
from pathlib import Path

import edge_tts
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

DATA = Path(os.getenv("DATA_DIR", "/tmp/recap"))
DATA.mkdir(parents=True, exist_ok=True)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # ပြောင်းလို့ရ
VOICES = {"thiha": "my-MM-ThihaNeural", "nilar": "my-MM-NilarNeural"}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = FastAPI()
jobs: dict[str, dict] = {}
gate = asyncio.Semaphore(1)  # တစ်ကြိမ်မှာ job ၁ ခုပဲ (CPU ကာကွယ်)

PROMPT = """You are a movie-recap narrator. Watch this short drama video.
Write a Burmese (Myanmar language) recap that retells the whole story in order,
engaging and dramatic like a movie-recap channel. Total narration should fit about
{target} seconds when spoken.
Return a JSON array. Each item has: "start" and "end" (seconds in the SOURCE video,
numbers, the scene that matches this narration, 3-15 seconds long, in increasing order),
and "narration" (1-2 Burmese sentences). Return only the JSON array."""


def run(cmd, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if p.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-400:]}")


def duration(path: Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(p.stdout.strip())


def plan(video: Path, target: int) -> list[dict]:
    f = client.files.upload(file=str(video))
    while f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError("Gemini could not process the video")
    r = client.models.generate_content(
        model=MODEL,
        contents=[f, PROMPT.format(target=target)],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.5),
    )
    segs = json.loads(r.text)
    return [s for s in segs if float(s["end"]) > float(s["start"]) and s["narration"].strip()]


def ts(sec: float) -> str:
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), sec % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


async def process(job_id: str, video: Path, voice: str, minutes: int, burn: bool):
    job, work = jobs[job_id], video.parent
    async with gate:
        try:
            job["status"] = "planning"
            segs = await asyncio.to_thread(plan, video, minutes * 60)
            parts, cues, t = [], [], 0.0
            for i, s in enumerate(segs):
                job["status"] = f"rendering {i + 1}/{len(segs)}"
                mp3, out = work / f"n{i}.mp3", work / f"p{i}.mp4"
                await edge_tts.Communicate(s["narration"], VOICES[voice]).save(str(mp3))
                dn = await asyncio.to_thread(duration, mp3)
                start, dc = float(s["start"]), float(s["end"]) - float(s["start"])
                k = min(2.0, max(0.5, dn / dc))  # ဗီဒီယိုကို စကားအရှည်နဲ့ ကိုက်အောင် ဆွဲ/ချုံ့
                await asyncio.to_thread(run, [
                    "ffmpeg", "-y", "-ss", str(start), "-t", str(dc), "-i", str(video), "-i", str(mp3),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", f"setpts={k}*PTS,tpad=stop_mode=clone:stop_duration=30,scale=-2:720,fps=30",
                    "-t", f"{dn:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-ar", "44100", "-ac", "2", str(out)])
                parts.append(out)
                cues.append(f"{i + 1}\n{ts(t)} --> {ts(t + dn)}\n{s['narration']}\n")
                t += dn
            job["status"] = "joining"
            (work / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
            joined = work / ("raw.mp4" if burn else "recap.mp4")
            await asyncio.to_thread(run, ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                          "-i", "list.txt", "-c", "copy", joined.name], str(work))
            if burn:
                (work / "cap.srt").write_text("\n".join(cues), encoding="utf-8")
                style = "FontName=Noto Sans Myanmar,FontSize=16,Outline=2,MarginV=40"
                await asyncio.to_thread(run, [
                    "ffmpeg", "-y", "-i", "raw.mp4", "-vf", f"subtitles=cap.srt:force_style='{style}'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "copy", "recap.mp4"], str(work))
            job["status"] = "done"
        except Exception as e:
            job["status"], job["error"] = "failed", str(e)[:500]


@app.post("/jobs")
async def create_job(bg: BackgroundTasks, file: UploadFile = File(...), voice: str = Form("thiha"),
                     minutes: int = Form(3), burn_captions: bool = Form(True)):
    if voice not in VOICES:
        raise HTTPException(400, "unknown voice")
    job_id = uuid.uuid4().hex[:12]
    work = DATA / job_id
    work.mkdir()
    video = work / "src.mp4"
    with video.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    jobs[job_id] = {"status": "queued"}
    bg.add_task(process, job_id, video, voice, minutes, burn_captions)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    f = DATA / job_id / "recap.mp4"
    if jobs.get(job_id, {}).get("status") != "done" or not f.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(f, media_type="video/mp4", filename="recap.mp4")
