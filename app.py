import os
import base64
import io
import numpy as np
import librosa
import requests
import tempfile
import subprocess
import soundfile as sf

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from joblib import load

# ---------------- CONFIG ----------------
SUPPORTED_LANGS = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

API_KEY = os.getenv("VOICE_API_KEY", "sk_test_123456789")
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

# Stricter AI decision to reduce false positives
AI_THRESHOLD = float(os.getenv("AI_THRESHOLD", "0.70"))

# Set False for final submission if you want generic errors only
DEBUG_ERRORS = True

# ---------------- LOAD MODEL ----------------
model = load(MODEL_PATH)
app = FastAPI(title="AI Voice Detection API")

# ---------------- SCHEMAS ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str | None = None
    audioUrl: str | None = None

# ---------------- AUDIO HELPERS ----------------
def decode_base64_mp3(b64: str, target_sr=16000, max_sec=6.0):
    raw = base64.b64decode(b64)

    # 1) Save MP3 to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_in:
        f_in.write(raw)
        in_path = f_in.name

    # 2) Convert MP3 -> WAV (16k mono) using ffmpeg
    out_path = in_path.replace(".mp3", ".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", str(target_sr),
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 3) Read wav safely
    y, sr = sf.read(out_path, dtype="float32")
    if y.ndim > 1:
        y = y[:, 0]

    # cleanup
    try:
        os.remove(in_path)
        os.remove(out_path)
    except:
        pass

    # pad/trim to fixed length
    max_len = int(target_sr * max_sec)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    return y, target_sr


def fetch_audio_url(url: str, timeout=20, max_bytes=3_000_000):
    """
    Download a public MP3 URL safely:
    - timeout
    - file size limit (prevents huge songs)
    - rejects HTML (Google Drive confirm pages)
    """
    r = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
    if r.status_code != 200:
        raise Exception(f"audioUrl fetch failed (status {r.status_code})")

    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        raise Exception("audioUrl returned HTML (not direct MP3). Use direct-download MP3 link.")

    data = b""
    for chunk in r.iter_content(chunk_size=65536):
        if not chunk:
            break
        data += chunk
        if len(data) > max_bytes:
            raise Exception("audioUrl file too large (use 5-10 sec voice mp3)")
    if not data:
        raise Exception("audioUrl returned empty content")
    return data

# ---------------- FEATURES (MUST MATCH TRAINING: 47 FEATURES) ----------------
def extract_features(y, sr):
    # 20 MFCC mean + 20 MFCC std = 40
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # +7 extra = 47 total
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    hf = S[freqs > 6000].mean() if np.any(freqs > 6000) else 0.0
    lf = S[(freqs > 300) & (freqs <= 6000)].mean()
    hf_ratio = float(hf / (lf + 1e-8))

    feat = np.concatenate([
        mfcc_mean, mfcc_std,
        [zcr, centroid, bandwidth, rolloff, rms, flatness, hf_ratio]
    ]).astype(np.float32)

    return feat

def build_explanation(feat):
    # For this 47-feature layout:
    # last 7 = [zcr, centroid, bandwidth, rolloff, rms, flatness, hf_ratio]
    zcr = float(feat[-7])
    flatness = float(feat[-2])
    hf_ratio = float(feat[-1])

    reasons = []
    if flatness > 0.20:
        reasons.append("Spectral flatness suggests synthetic uniformity")
    if hf_ratio > 0.30:
        reasons.append("High-frequency artifacts are unusually strong")
    if zcr < 0.04:
        reasons.append("Low micro-variation patterns detected")

    if not reasons:
        reasons.append("Natural variation consistent with human speech")
    return "; ".join(reasons[:2])

# ---------------- ENDPOINT ----------------
@app.post("/api/voice-detection")
def voice_detection(req: VoiceRequest, x_api_key: str = Header(None)):
    # API Key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={
            "status": "error",
            "message": "Invalid API key or malformed request"
        })

    # Validate input fields
    if req.language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "Unsupported language"
        })

    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "audioFormat must be mp3"
        })

    if not req.audioBase64 and not req.audioUrl:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "Invalid API key or malformed request"
        })

    try:
        # If audioUrl provided, fetch MP3 bytes -> base64
        if req.audioUrl and not req.audioBase64:
            audio_bytes = fetch_audio_url(req.audioUrl)
            req.audioBase64 = base64.b64encode(audio_bytes).decode()

        # Decode base64 MP3 -> waveform
        y, sr = decode_base64_mp3(req.audioBase64)

        # Extract 47 features -> model
        feat = extract_features(y, sr).reshape(1, -1)

        proba_human = float(model.predict_proba(feat)[0][1])
        proba_ai = 1.0 - proba_human

        # Strict threshold
        if proba_ai >= AI_THRESHOLD:
            classification = "AI_GENERATED"
            confidence = proba_ai
        else:
            classification = "HUMAN"
            confidence = proba_human

        return {
            "status": "success",
            "language": req.language,
            "classification": classification,
            "confidenceScore": round(float(confidence), 4),
            "explanation": build_explanation(feat[0])
        }

    except Exception as e:
        msg = f"DEBUG: {type(e).__name__}: {str(e)}" if DEBUG_ERRORS else "Invalid API key or malformed request"
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": msg
        })

