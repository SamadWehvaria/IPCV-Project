"""
AI Examination System — Streamlit Frontend
Integrates: Handwritten Test (OCR/Vision), Typing Test (paste detection + AI detection),
Online Meeting Test (video transcription + CV proctoring), Teacher Dashboard.
"""

import streamlit as st
import os, json, re, time, warnings
import numpy as np
from collections import deque
from PIL import Image, ImageDraw
import cv2
import torch
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI ExamSystem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --panel: #13161e;
    --border: #1e2330;
    --accent: #4fffb0;
    --accent2: #ff6b6b;
    --accent3: #6b8eff;
    --text: #e8ecf0;
    --muted: #6b7280;
    --warn: #fbbf24;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main area */
.main .block-container { padding: 2rem 2.5rem; max-width: 1100px; }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.02em; }
h1 { font-size: 2rem !important; color: var(--accent) !important; }
h2 { font-size: 1.3rem !important; color: var(--text) !important; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
h3 { font-size: 1rem !important; color: var(--accent3) !important; }

/* Cards */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* Metric badges */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    flex: 1; min-width: 120px;
}
.metric-box .val { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: var(--accent); }
.metric-box .lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* Status pills */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.pill-ok   { background: #1a2e24; color: var(--accent);  border: 1px solid var(--accent); }
.pill-warn { background: #2e1f0a; color: var(--warn);   border: 1px solid var(--warn); }
.pill-err  { background: #2e1212; color: var(--accent2); border: 1px solid var(--accent2); }

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* File uploader */
.stFileUploader { background: var(--panel) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important; }

/* Expander */
.streamlit-expanderHeader { font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important; }

/* Alert boxes */
.stAlert { border-radius: 8px !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Logo area */
.logo-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1.2rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.logo-bar .logo-icon { font-size: 1.8rem; }
.logo-bar .logo-text { font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; color: var(--accent); }
.logo-bar .logo-sub  { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "models_loaded": False,
        "nlp_model": None,
        "ocr_reader": None,
        "whisper_model": None,
        "ai_detector": None,
        "results_db": {},
        "page": "home",
        "handwritten_exam": {
            "1": {"question": "What is machine learning?",
                  "answer": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                  "marks": 10},
            "2": {"question": "Define supervised learning.",
                  "answer": "Supervised learning is a type of machine learning where the model is trained on labeled data, meaning each training example has an input and a known correct output.",
                  "marks": 10},
        },
        "typing_exam": {
            "1": {"question": "What is machine learning?",
                  "answer": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                  "marks": 10},
        },
        "online_exam": {
            "1": {"question": "What is machine learning?",
                  "answer": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                  "marks": 10},
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from sentence_transformers import SentenceTransformer
    import easyocr
    import whisper
    from transformers import pipeline as hf_pipeline

    nlp_model     = SentenceTransformer("all-MiniLM-L6-v2")
    ocr_reader    = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    whisper_model = whisper.load_model("base")
    ai_detector   = hf_pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        device=0 if torch.cuda.is_available() else -1,
    )
    return nlp_model, ocr_reader, whisper_model, ai_detector

# ─────────────────────────────────────────────────────────
# CORE LOGIC (mirrors notebook)
# ─────────────────────────────────────────────────────────
def compute_similarity(student_answer: str, model_answer: str) -> float:
    from sentence_transformers import util
    nlp = st.session_state.nlp_model
    if not student_answer.strip() or nlp is None:
        return 0.0
    emb1 = nlp.encode(student_answer, convert_to_tensor=True)
    emb2 = nlp.encode(model_answer, convert_to_tensor=True)
    score = float(util.cos_sim(emb1, emb2)[0][0])
    return round(max(0.0, min(1.0, score)), 3)


def score_answers(student_answers: dict, answer_key: dict) -> dict:
    results = {}
    total_scored = total_marks = 0
    for q_id, meta in answer_key.items():
        student_ans = student_answers.get(q_id, "").strip()
        model_ans   = meta["answer"]
        max_marks   = meta.get("marks", 10)
        student_clean = student_ans.lower()

        if not student_ans or student_clean in ["(no answer)", "no answer", "n/a", "-", "none", ""]:
            scored, feedback, sim = 0, "❌ No answer provided", 0.0
        else:
            sim = compute_similarity(student_ans, model_ans)
            if sim >= 0.5:
                scored   = max_marks
                feedback = "✅ Correct concept"
            elif sim >= 0.3:
                scored   = round(max_marks * 0.7, 1)
                feedback = "🟡 Partially correct"
            else:
                keywords = {"machine","learning","artificial","intelligence","learn","data","experience","programmed"}
                matches  = sum(1 for kw in keywords if kw in student_ans.lower())
                if matches >= 3:
                    scored   = round(max_marks * 0.6, 1)
                    feedback = "🟠 Has key concepts"
                else:
                    scored   = round(max_marks * 0.2, 1)
                    feedback = "❌ Mostly incorrect"

        total_scored += scored
        total_marks  += max_marks
        results[q_id] = {
            "question":    meta["question"],
            "student_ans": student_ans or "(no answer)",
            "model_ans":   model_ans,
            "similarity":  sim,
            "score":       scored,
            "max_marks":   max_marks,
            "feedback":    feedback,
        }
    results["__total__"] = {"scored": round(total_scored, 2), "max": total_marks}
    return results


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    blur     = cv2.GaussianBlur(denoised, (3, 3), 0)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def extract_text_ocr(pil_image: Image.Image) -> str:
    reader = st.session_state.ocr_reader
    if reader is None:
        return ""
    processed = preprocess_image(pil_image)
    img_np = np.array(processed)
    results = reader.readtext(img_np, detail=0, paragraph=True, width_ths=0.7, height_ths=0.7, text_threshold=0.6)
    return " ".join(results)


def parse_handwritten_answers(raw_text: str, exam: dict) -> dict:
    answers = {}
    pattern = r"(?:Q|Question|Ans(?:wer)?)?\s*([0-9]+)[\.:\)]"
    splits  = re.split(pattern, raw_text, flags=re.IGNORECASE)
    i, found = 1, False
    while i < len(splits) - 1:
        q_num  = splits[i].strip()
        q_text = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if q_num in exam:
            answers[q_num] = q_text
            found = True
        i += 2
    if not found:
        q_ids = list(exam.keys())
        chunk = max(1, len(raw_text) // len(q_ids))
        for idx, q_id in enumerate(q_ids):
            answers[q_id] = raw_text[idx * chunk:(idx + 1) * chunk].strip()
    for q_id in exam:
        if q_id not in answers:
            answers[q_id] = ""
    return answers


PASTE_THRESHOLD = 20


import re
from spellchecker import SpellChecker
from difflib import SequenceMatcher

spell = SpellChecker()

def extract_student_info(text: str) -> dict:
    """Extract Name and Roll Number from OCR text"""
    info = {'name': None, 'roll_no': None}
    
    # Roll No patterns
    roll_patterns = [
        r'Roll\s*(?:No|Number|#)?[:.\s]*(\d{4,12})',
        r'Roll[:.\s]*(\d{4,12})',
        r'(\d{6,12})\s*(?:Roll|ID|Enrollment)',
        r'\b(\d{6,12})\b'
    ]
    for pat in roll_patterns:
        m = re.search(pat, text, re.I)
        if m:
            info['roll_no'] = m.group(1)
            break
    
    # Name patterns
    name_patterns = [
        r'Name[:.\s]*(.+?)(?=Roll|Class|Date|Q\d|\n|$)',
        r'Name[:.\s]*([A-Z][a-zA-Z\s]{5,30})',
        r'^([A-Z][a-zA-Z\s]{5,})'
    ]
    for pat in name_patterns:
        m = re.search(pat, text, re.I | re.M)
        if m and len(m.group(1).strip().split()) >= 2:
            info['name'] = m.group(1).strip()
            break
            
    return info


def correct_spelling(text: str):
    """Simple spelling correction with penalty"""
    if not text or len(text) < 15:
        return text, 1.0
    
    words = text.split()
    corrected = []
    corrections = 0
    for word in words:
        if word.isalpha() and len(word) > 3:
            corr = spell.correction(word)
            if corr and corr.lower() != word.lower():
                corrections += 1
                corrected.append(corr)
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    
    corrected_text = " ".join(corrected)
    penalty = min(0.18, corrections * 0.025)   # max ~18% penalty
    return corrected_text, (1.0 - penalty)


def score_with_spellcheck(student_answer: str, model_answer: str, max_marks: int = 10):
    """Enhanced scoring with spelling awareness"""
    if not student_answer.strip():
        return 0.0, "No answer provided", 0.0
    
    corrected, spell_factor = correct_spelling(student_answer)
    
    # Semantic similarity
    sim = compute_similarity(corrected, model_answer)
    
    # Final score = 70% semantic + 30% spelling
    final_score = (sim * 0.7 + spell_factor * 0.3) * max_marks
    
    if final_score >= max_marks * 0.85:
        feedback = "✅ Excellent (Good spelling & content)"
    elif final_score >= max_marks * 0.65:
        feedback = "🟡 Good, minor spelling issues"
    elif final_score >= max_marks * 0.45:
        feedback = "🟠 Partially correct, improve spelling"
    else:
        feedback = "❌ Needs major improvement"
    
    return round(final_score, 1), feedback, sim



def analyze_text_deltas(delta_log: list) -> dict:
    paste_events = [d for d in delta_log if d >= PASTE_THRESHOLD]
    return {
        "copy_paste_detected": len(paste_events) > 0,
        "paste_event_count":   len(paste_events),
        "largest_paste":       max(paste_events, default=0),
        "details":             [f"Paste event: +{d} characters inserted at once" for d in paste_events],
    }


def detect_ai_content(text: str) -> dict:
    detector = st.session_state.ai_detector
    if detector is None or not text.strip() or len(text.split()) < 10:
        return {"ai_generated": False, "confidence": 0.0, "label": "Models not loaded / too short"}
    try:
        result = detector(text[:512])[0]
        label  = result["label"]
        score  = round(result["score"], 3)
        is_ai  = (label == "LABEL_1") and (score > 0.7)
        return {"ai_generated": is_ai, "confidence": score,
                "label": f"{'AI-generated' if is_ai else 'Human-written'} ({score*100:.1f}% confidence)"}
    except Exception as e:
        return {"ai_generated": False, "confidence": 0.0, "label": f"Detection error: {e}"}


def transcribe_audio(audio_path: str) -> str:
    wm = st.session_state.whisper_model
    if wm is None:
        return ""
    result = wm.transcribe(audio_path)
    return result["text"].strip()


def parse_spoken_answers(transcript: str, exam: dict) -> dict:
    answers = {}
    num_words = {"1":"one","2":"two","3":"three","4":"four","5":"five",
                 "6":"six","7":"seven","8":"eight","9":"nine","10":"ten"}
    lower = transcript.lower()
    breakpoints = {}
    for q_id, word in num_words.items():
        if q_id not in exam:
            continue
        for pat in [f"question {word}", f"question {q_id}",
                    f"answer {word}", f"answer {q_id}", f"q{q_id}"]:
            idx = lower.find(pat)
            if idx != -1:
                breakpoints[q_id] = idx
                break
    if len(breakpoints) >= 2:
        sorted_bps = sorted(breakpoints.items(), key=lambda x: x[1])
        for i, (q_id, start) in enumerate(sorted_bps):
            end = sorted_bps[i+1][1] if i+1 < len(sorted_bps) else len(transcript)
            answers[q_id] = transcript[start:end].strip()
    else:
        q_ids = list(exam.keys())
        chunk = max(1, len(transcript) // len(q_ids))
        for idx, q_id in enumerate(q_ids):
            answers[q_id] = transcript[idx*chunk:(idx+1)*chunk].strip()
    for q_id in exam:
        if q_id not in answers:
            answers[q_id] = ""
    return answers


def analyze_video_proctoring(video_path: str) -> dict:
    """Simplified CV proctoring — runs even without mediapipe."""
    events      = []
    face_counts = []

    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh    = mp_face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        LEFT_EYE  = [362,385,387,263,373,380]
        RIGHT_EYE = [33,160,158,133,153,144]
        MOUTH     = [78,308,13,14]
        MODEL_PTS = np.array([(0,0,0),(-225,170,-135),(225,170,-135),
                               (-150,-150,-125),(150,-150,-125)], dtype=np.float32)
        EAR_THRESHOLD = 0.25; MAR_THRESHOLD = 0.5
        YAW_THRESHOLD = 35;   PITCH_THRESHOLD = 25
        EMA_ALPHA = 0.35
        ear_buf   = deque(maxlen=12)
        mar_buf   = deque(maxlen=12)
        gaze_buf  = deque(maxlen=12)
        ema_pitch = ema_yaw = None

        def calc_ear(lm, eye, w, h):
            pts = [np.array([lm[i].x*w, lm[i].y*h]) for i in eye]
            return (np.linalg.norm(pts[1]-pts[5])+np.linalg.norm(pts[2]-pts[4]))/(2*np.linalg.norm(pts[0]-pts[3])+1e-6)

        def calc_mar(lm, mouth, w, h):
            pts = [np.array([lm[i].x*w, lm[i].y*h]) for i in mouth]
            return np.linalg.norm(pts[2]-pts[3])/(np.linalg.norm(pts[0]-pts[1])+1e-6)

        def head_pose(lm, w, h):
            nonlocal ema_pitch, ema_yaw
            img_pts = np.array([(lm[1].x*w,lm[1].y*h),(lm[152].x*w,lm[152].y*h),
                                 (lm[33].x*w,lm[33].y*h),(lm[263].x*w,lm[263].y*h),
                                 (lm[61].x*w,lm[61].y*h)], dtype=np.float32)
            focal = w
            cam_m = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float32)
            ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam_m, np.zeros((4,1)), flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return ema_pitch or 0, ema_yaw or 0
            rmat, _ = cv2.Rodrigues(rvec)
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(np.hstack((rmat, np.zeros((3,1)))))
            rp = float(euler[0][0]); ry = -float(euler[1][0])
            if ema_pitch is None: ema_pitch, ema_yaw = rp, ry
            else:
                ema_pitch = EMA_ALPHA*rp+(1-EMA_ALPHA)*ema_pitch
                ema_yaw   = EMA_ALPHA*ry+(1-EMA_ALPHA)*ema_yaw
            return ema_pitch, ema_yaw

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            if idx % 3 != 0: continue
            ts = round(idx/fps, 2)
            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = face_mesh.process(rgb)
            fc   = len(res.multi_face_landmarks) if res.multi_face_landmarks else 0
            face_counts.append(fc)
            if fc == 0:
                events.append((ts, "NO_FACE", "Student not visible"))
                gaze_buf.append(True)
            elif fc >= 2:
                events.append((ts, "MULTIPLE_FACES", f"{fc} people in frame"))
            if res.multi_face_landmarks:
                for fl in res.multi_face_landmarks:
                    lm = fl.landmark
                    # Too close check
                    lc = np.array([lm[33].x*w, lm[33].y*h])
                    rc = np.array([lm[263].x*w, lm[263].y*h])
                    fw = np.linalg.norm(rc-lc)
                    if fw > w*0.45:
                        events.append((ts, "TOO_CLOSE", f"Face width:{fw:.0f}px"))
                    # Head pose
                    pitch, yaw = head_pose(lm, w, h)
                    distracted = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD
                    gaze_buf.append(distracted)
                    if sum(gaze_buf) >= int(0.7*len(gaze_buf)):
                        events.append((ts, "GAZE_AWAY", f"Yaw:{yaw:.1f}° Pitch:{pitch:.1f}°"))
                    # EAR / MAR
                    ear = (calc_ear(lm,LEFT_EYE,w,h)+calc_ear(lm,RIGHT_EYE,w,h))/2
                    mar = calc_mar(lm,MOUTH,w,h)
                    ear_buf.append(ear < EAR_THRESHOLD)
                    mar_buf.append(mar > MAR_THRESHOLD)
                    if sum(ear_buf) >= int(0.8*len(ear_buf)):
                        events.append((ts, "DROWSY", ""))
                    if sum(mar_buf) >= int(0.6*len(mar_buf)):
                        events.append((ts, "YAWNING", ""))
        cap.release()
    except ImportError:
        # mediapipe not available — use OpenCV face detection fallback
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            if idx % 5 != 0: continue
            ts   = round(idx/fps, 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            fc = len(faces)
            face_counts.append(fc)
            if fc == 0:
                events.append((ts, "NO_FACE", "Student not visible"))
            elif fc >= 2:
                events.append((ts, "MULTIPLE_FACES", f"{fc} people in frame"))
        cap.release()

    avg_faces = float(np.mean(face_counts)) if face_counts else 0
    gaze_aways = len([e for e in events if e[1] == "GAZE_AWAY"])
    return {
        "cheating_detected":    avg_faces < 0.7 or any(c >= 2 for c in face_counts) or gaze_aways > 20,
        "avg_faces":            round(avg_faces, 2),
        "no_face_events":       len([e for e in events if e[1] == "NO_FACE"]),
        "multiple_face_events": len([e for e in events if e[1] == "MULTIPLE_FACES"]),
        "gaze_away_events":     gaze_aways,
        "eyes_down_events":     len([e for e in events if e[1] == "EYES_DOWN"]),
        "too_close_events":     len([e for e in events if e[1] == "TOO_CLOSE"]),
        "drowsy_events":        len([e for e in events if e[1] == "DROWSY"]),
        "phone_detected":       False,
        "details":              [f"[{t:.1f}s] {typ}: {msg}" for t,typ,msg in events[:40]],
    }


def extract_audio_from_video(video_path: str, out: str = "/tmp/exam_audio.wav") -> str:
    import moviepy.editor as mpy
    clip = mpy.VideoFileClip(video_path)
    clip.audio.write_audiofile(out, verbose=False, logger=None)
    return out


def analyze_audio_proctoring(audio_path: str) -> dict:
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        events = []
        frame_length = sr // 2
        for i in range(0, len(y) - frame_length, frame_length):
            chunk = y[i:i+frame_length]
            rms   = float(np.sqrt(np.mean(chunk**2)))
            ts    = round(i/sr, 1)
            if rms > 0.08:
                events.append((ts, "AUDIO_SPIKE", f"RMS:{rms:.3f}"))
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        active = pitches[pitches > 0]
        multi  = False
        if len(active) > 0 and np.std(active) > 800:
            multi = True
            events.append((0.0, "MULTIPLE_SPEAKERS", f"Pitch std:{np.std(active):.0f}"))
        return {
            "audio_spike_events": len([e for e in events if e[1]=="AUDIO_SPIKE"]),
            "multiple_speakers":  multi,
            "details":            [f"[{t:.1f}s] {typ}: {msg}" for t,typ,msg in events],
        }
    except Exception:
        return {"audio_spike_events": 0, "multiple_speakers": False, "details": []}

# ─────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────
def pill(label: str, kind: str = "ok") -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def show_score_card(results: dict):
    total = results.get("__total__", {"scored": 0, "max": 0})
    pct   = round(total["scored"] / max(total["max"], 1) * 100, 1)
    color = "#4fffb0" if pct >= 70 else "#fbbf24" if pct >= 40 else "#ff6b6b"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="val" style="color:{color}">{total['scored']}<span style="font-size:1rem;color:#6b7280">/{total['max']}</span></div>
        <div class="lbl">TOTAL SCORE</div>
      </div>
      <div class="metric-box">
        <div class="val" style="color:{color}">{pct}%</div>
        <div class="lbl">PERCENTAGE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)
    for q_id, r in results.items():
        if q_id == "__total__": continue
        with st.expander(f"Q{q_id}: {r['question'][:60]}…  —  {r['score']}/{r['max_marks']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Student Answer**")
                st.info(r["student_ans"] or "(no answer)")
            with col2:
                st.markdown("**Model Answer**")
                st.success(r["model_ans"])
            sim_pct = int(r["similarity"] * 100)
            st.markdown(f"Similarity: **{sim_pct}%** — {r['feedback']}")
            st.progress(r["similarity"])


def show_cheating_report(cr: dict):
    if not cr:
        st.markdown(pill("No proctoring data", "warn"), unsafe_allow_html=True)
        return
    flags = []
    if cr.get("copy_paste_detected"):    flags.append("Copy-paste")
    if cr.get("ai_content_detected"):    flags.append("AI content")
    if cr.get("cheating_detected"):      flags.append("CV cheating")
    if cr.get("multiple_speakers"):      flags.append("Multiple speakers")
    if cr.get("phone_detected"):         flags.append("Phone detected")

    if flags:
        st.markdown(" ".join(pill(f, "err") for f in flags), unsafe_allow_html=True)
    else:
        st.markdown(pill("✅ Clean", "ok"), unsafe_allow_html=True)

    details = cr.get("details", [])
    if details:
        with st.expander("Full event log", expanded=False):
            for d in details:
                st.code(d, language=None)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
      <div class="logo-icon">🎓</div>
      <div>
        <div class="logo-text">AI ExamSystem</div>
        <div class="logo-sub">Intelligent Examination Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", ["🏠 Home", "⚙️ Setup Models", "📋 Manage Exam",
                                   "✍️ Handwritten Test", "⌨️ Typing Test",
                                   "🎥 Online Meeting Test", "📊 Teacher Dashboard"],
                   label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.75rem;color:#6b7280;'>Results stored: <b>{len(st.session_state.results_db)}</b> student(s)</div>", unsafe_allow_html=True)
    loaded = st.session_state.models_loaded
    st.markdown(pill("Models loaded" if loaded else "Models not loaded", "ok" if loaded else "warn"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────

# ── HOME ──────────────────────────────────────────────────
if nav == "🏠 Home":
    st.title("AI Examination System")
    st.markdown("*OCR · NLP · Computer Vision · Behavioral Analytics*")
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="card"><h3>✍️ Handwritten</h3><p style="color:#6b7280;font-size:0.9rem;">Upload scanned answer sheets. OCR extracts text, NLP scores against answer key with semantic similarity.</p></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="card"><h3>⌨️ Typing Test</h3><p style="color:#6b7280;font-size:0.9rem;">Real-time copy-paste detection + AI-generated content analysis. Every keystroke monitored.</p></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="card"><h3>🎥 Online Meeting</h3><p style="color:#6b7280;font-size:0.9rem;">Upload recorded video. Whisper transcribes speech, MediaPipe tracks face, gaze, and head pose.</p></div>', unsafe_allow_html=True)

    st.markdown("### Quick Start")
    st.markdown("""
1. Go to **⚙️ Setup Models** and click *Load Models* (required once per session)  
2. Go to **📋 Manage Exam** to define your questions and answer keys  
3. Choose a test type and run it as a student  
4. View all results in **📊 Teacher Dashboard**
    """)

# ── SETUP MODELS ──────────────────────────────────────────
elif nav == "⚙️ Setup Models":
    st.title("Model Setup")
    st.markdown("Load all required AI models. This may take 1–3 minutes on first run.")
    st.markdown("""
    **Models loaded:**
    - `all-MiniLM-L6-v2` — Sentence Transformers for NLP scoring
    - `EasyOCR` — Handwritten text extraction
    - `Whisper base` — Speech-to-text for online tests
    - `roberta-base-openai-detector` — AI-generated content detection
    """)

    if st.session_state.models_loaded:
        st.success("✅ All models are loaded and ready.")
    else:
        if st.button("🚀 Load All Models"):
            with st.spinner("Loading models — please wait…"):
                try:
                    nlp, ocr, wsp, ai_det = load_models()
                    st.session_state.nlp_model    = nlp
                    st.session_state.ocr_reader   = ocr
                    st.session_state.whisper_model = wsp
                    st.session_state.ai_detector  = ai_det
                    st.session_state.models_loaded = True
                    st.success("✅ All models loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    st.info("Make sure all dependencies are installed: easyocr, sentence-transformers, openai-whisper, transformers")

# ── MANAGE EXAM ───────────────────────────────────────────
elif nav == "📋 Manage Exam":
    st.title("Exam Question Manager")
    tab1, tab2, tab3 = st.tabs(["✍️ Handwritten", "⌨️ Typing", "🎥 Online Meeting"])

    def exam_editor(key: str, label: str):
        exam = st.session_state[key]
        st.markdown(f"**Current questions ({len(exam)})**")
        for q_id, meta in list(exam.items()):
            with st.expander(f"Q{q_id}: {meta['question'][:50]}…", expanded=False):
                new_q = st.text_input("Question", meta["question"], key=f"{key}_q_{q_id}")
                new_a = st.text_area("Model Answer", meta["answer"], key=f"{key}_a_{q_id}", height=80)
                new_m = st.number_input("Marks", 1, 100, meta["marks"], key=f"{key}_m_{q_id}")
                col1, col2 = st.columns([1, 3])
                if col1.button("💾 Save", key=f"{key}_save_{q_id}"):
                    st.session_state[key][q_id] = {"question": new_q, "answer": new_a, "marks": new_m}
                    st.success("Saved!")
                if col2.button("🗑 Delete", key=f"{key}_del_{q_id}"):
                    del st.session_state[key][q_id]
                    st.rerun()

        st.markdown("---")
        st.markdown("**Add new question**")
        new_id = str(len(exam) + 1)
        nq = st.text_input("Question", key=f"{key}_new_q")
        na = st.text_area("Model Answer", key=f"{key}_new_a", height=80)
        nm = st.number_input("Marks", 1, 100, 10, key=f"{key}_new_m")
        if st.button("➕ Add Question", key=f"{key}_add"):
            if nq and na:
                st.session_state[key][new_id] = {"question": nq, "answer": na, "marks": nm}
                st.success(f"Question {new_id} added!")
                st.rerun()
            else:
                st.warning("Please fill in both question and answer.")

    with tab1: exam_editor("handwritten_exam", "Handwritten")
    with tab2: exam_editor("typing_exam",      "Typing")
    with tab3: exam_editor("online_exam",      "Online Meeting")


# ── HANDWRITTEN TEST ──────────────────────────────────────
elif nav == "✍️ Handwritten Test":
    st.title("✍️ Handwritten Answer Sheet Evaluator")

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first.")
        st.stop()

    exam = st.session_state.handwritten_exam
    if not exam:
        st.warning("No questions defined. Go to **Manage Exam**.")
        st.stop()

    student_name = st.text_input("Full Name", placeholder="e.g. Muhammad Ahmed")

    st.markdown("**Upload Answer Sheet**")
    uploaded = st.file_uploader("Scanned Answer Sheet (PNG/JPG)", 
                                type=["png","jpg","jpeg"], key="hw_upload")

    if uploaded:
        pil_img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original", use_container_width=True)
        with col2:
            processed = preprocess_image(pil_img)
            st.image(processed, caption="Preprocessed", use_container_width=True)

    if st.button("📤 Submit & Evaluate", disabled=not (student_name and uploaded), type="primary"):
        with st.spinner("Preprocessing → OCR → Scoring..."):
            pil_img = Image.open(uploaded)
            
            # Enhanced OCR
            raw_text = extract_text_ocr(pil_img)
            
            # Extract Name & Roll No
            student_info = extract_student_info(raw_text)
            
            st.markdown("**Extracted Text:**")
            st.code(raw_text[:800] + "..." if len(raw_text) > 800 else raw_text, language=None)
            
            st.info(f"**Detected Student Info** → Name: **{student_info['name'] or 'Not Found'}** | Roll No: **{student_info['roll_no'] or 'Not Found'}**")

            # Enhanced Scoring
            student_ans = parse_handwritten_answers(raw_text, exam)
            score_report = {}

            total_scored = total_max = 0
            for q_id, meta in exam.items():
                ans_text = student_ans.get(q_id, raw_text)  # fallback
                score, feedback, sim = score_with_spellcheck(
                    ans_text, meta["answer"], meta.get("marks", 10)
                )
                score_report[q_id] = {
                    "question": meta["question"],
                    "student_ans": ans_text[:500],
                    "model_ans": meta["answer"],
                    "similarity": sim,
                    "score": score,
                    "max_marks": meta.get("marks", 10),
                    "feedback": feedback,
                }
                total_scored += score
                total_max += meta.get("marks", 10)

            score_report["__total__"] = {"scored": round(total_scored, 1), "max": total_max}

            # Save result
            if student_name not in st.session_state.results_db:
                st.session_state.results_db[student_name] = {}
            st.session_state.results_db[student_name]["Handwritten Test"] = {
                "score_report": score_report,
                "student_info": student_info,
                "raw_text": raw_text[:1000],
                "cheating_report": {"note": "Offline handwritten - No behavioral proctoring"}
            }

            st.success("✅ Evaluation Complete!")
            show_score_card(score_report)

# ── TYPING TEST ───────────────────────────────────────────
elif nav == "⌨️ Typing Test":
    st.title("Typing Test — Behavioral Monitoring")

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first via **⚙️ Setup Models**.")
        st.stop()

    exam = st.session_state.typing_exam
    if not exam:
        st.warning("No questions defined. Go to **📋 Manage Exam** to add questions.")
        st.stop()

    st.info("ℹ️ Type your answers below. Large text insertions (≥ 20 chars at once) are flagged as paste events. AI detection runs on submission.")

    student_name = st.text_input("Full Name", placeholder="e.g. Ahmed Khan", key="typing_name")

    # Initialize answer storage in session state
    if "typing_answers" not in st.session_state:
        st.session_state.typing_answers = {q_id: "" for q_id in exam}
    if "typing_prev" not in st.session_state:
        st.session_state.typing_prev = {q_id: "" for q_id in exam}
    if "typing_deltas" not in st.session_state:
        st.session_state.typing_deltas = {q_id: [] for q_id in exam}
    if "typing_warnings" not in st.session_state:
        st.session_state.typing_warnings = {q_id: "" for q_id in exam}

    # Sync keys in case exam changed
    for q_id in exam:
        st.session_state.typing_answers.setdefault(q_id, "")
        st.session_state.typing_prev.setdefault(q_id, "")
        st.session_state.typing_deltas.setdefault(q_id, [])
        st.session_state.typing_warnings.setdefault(q_id, "")

    for q_id, meta in exam.items():
        st.markdown(f"### Q{q_id} ({meta['marks']} marks)")
        st.markdown(f"*{meta['question']}*")

        def make_change_handler(qid):
            def on_change():
                new_text = st.session_state[f"typing_box_{qid}"]
                prev     = st.session_state.typing_prev[qid]
                delta    = len(new_text) - len(prev)
                st.session_state.typing_answers[qid] = new_text
                st.session_state.typing_prev[qid]    = new_text
                st.session_state.typing_deltas[qid].append(delta)
                if delta >= PASTE_THRESHOLD:
                    st.session_state.typing_warnings[qid] = (
                        f"⚠️ **Paste detected!** +{delta} characters inserted at once. This has been logged."
                    )
                else:
                    st.session_state.typing_warnings[qid] = ""
            return on_change

        st.text_area(
            f"Answer {q_id}",
            value=st.session_state.typing_answers[q_id],
            key=f"typing_box_{q_id}",
            height=120,
            placeholder="Type your answer here (do NOT paste)…",
            on_change=make_change_handler(q_id),
        )
        warn = st.session_state.typing_warnings[q_id]
        if warn:
            st.warning(warn)
        st.markdown("---")

    if st.button("📤 Submit Typing Test", disabled=not student_name):
        if not student_name:
            st.error("Please enter student name.")
        else:
            with st.spinner("Analyzing answers and detecting cheating…"):
                student_answers = {q_id: st.session_state.typing_answers.get(q_id, "")
                                   for q_id in exam}
                paste_reports = {q_id: analyze_text_deltas(st.session_state.typing_deltas.get(q_id, []))
                                 for q_id in exam}
                ai_reports    = {q_id: detect_ai_content(student_answers[q_id])
                                 for q_id in exam}
                score_report  = score_answers(student_answers, exam)

                any_paste = any(r["copy_paste_detected"] for r in paste_reports.values())
                any_ai    = any(r["ai_generated"]        for r in ai_reports.values())
                details   = []
                for q_id in exam:
                    pr = paste_reports[q_id]
                    ar = ai_reports[q_id]
                    if pr["copy_paste_detected"]:
                        details.append(f"Q{q_id}: {pr['paste_event_count']} paste event(s) — largest +{pr['largest_paste']} chars")
                    if ar["ai_generated"]:
                        details.append(f"Q{q_id}: AI content — {ar['label']}")

                cheating_report = {
                    "copy_paste_detected": any_paste,
                    "ai_content_detected": any_ai,
                    "details":             details,
                }
                if student_name not in st.session_state.results_db:
                    st.session_state.results_db[student_name] = {}
                st.session_state.results_db[student_name]["Typing Test"] = {
                    "score_report":    score_report,
                    "cheating_report": cheating_report,
                }

            st.success("✅ Submitted successfully!")
            st.markdown("## 📋 Score Report")
            show_score_card(score_report)
            st.markdown("## 🔍 Integrity Report")
            show_cheating_report(cheating_report)

            # Reset state
            for k in ["typing_answers","typing_prev","typing_deltas","typing_warnings"]:
                del st.session_state[k]

# ── ONLINE MEETING TEST ───────────────────────────────────
elif nav == "🎥 Online Meeting Test":
    st.title("Online Meeting Test — Video Proctoring")

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first via **⚙️ Setup Models**.")
        st.stop()

    exam = st.session_state.online_exam
    if not exam:
        st.warning("No questions defined. Go to **📋 Manage Exam** to add questions.")
        st.stop()

    st.markdown("""
    Upload a recorded exam video. The system will:
    - **Transcribe** speech using Whisper
    - **Score** answers with NLP semantic similarity
    - **Proctor** video for face visibility, gaze direction, multiple faces, and suspicious behavior
    - **Analyze audio** for background voices and anomalies
    """)

    student_name = st.text_input("Full Name", placeholder="e.g. Ahmed Khan", key="online_name")
    video_file   = st.file_uploader("Upload Exam Video (MP4, AVI, MOV)", type=["mp4","avi","mov","mkv"])

    if video_file:
        st.video(video_file)

    st.markdown("**Questions in this exam:**")
    for q_id, meta in exam.items():
        st.markdown(f"- **Q{q_id}** ({meta['marks']} marks): {meta['question']}")

    if st.button("📤 Submit Video for Analysis", disabled=not (student_name and video_file)):
        if not student_name or not video_file:
            st.error("Please enter student name and upload a video.")
        else:
            # Save video to temp file
            tmp_path = f"/tmp/exam_video_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

            col1, col2, col3 = st.columns(3)
            prog = st.progress(0, text="Starting analysis…")

            with st.spinner("Step 1/3 — Transcribing audio…"):
                try:
                    audio_path = extract_audio_from_video(tmp_path)
                    transcript = transcribe_audio(audio_path)
                    prog.progress(33, text="Audio transcribed ✓")
                except Exception as e:
                    audio_path = None
                    transcript = ""
                    st.warning(f"Audio extraction failed: {e}")

            spoken_answers = parse_spoken_answers(transcript, exam)
            score_report   = score_answers(spoken_answers, exam)

            with st.spinner("Step 2/3 — Running video proctoring…"):
                cheating_report = analyze_video_proctoring(tmp_path)
                prog.progress(66, text="Video proctoring done ✓")

            with st.spinner("Step 3/3 — Analyzing audio proctoring…"):
                if audio_path:
                    audio_report = analyze_audio_proctoring(audio_path)
                    cheating_report["audio_spike_events"] = audio_report["audio_spike_events"]
                    cheating_report["multiple_speakers"]  = audio_report["multiple_speakers"]
                    cheating_report["details"]           += audio_report["details"]
                    if audio_report["multiple_speakers"] or audio_report["audio_spike_events"] > 5:
                        cheating_report["cheating_detected"] = True
                prog.progress(100, text="Analysis complete ✓")

            if student_name not in st.session_state.results_db:
                st.session_state.results_db[student_name] = {}
            st.session_state.results_db[student_name]["Online Test"] = {
                "score_report":    score_report,
                "cheating_report": cheating_report,
            }

            # Clean up
            try: os.remove(tmp_path)
            except: pass

            st.success("✅ Analysis complete!")
            st.markdown("---")

            with st.expander("📜 Transcript", expanded=True):
                st.code(transcript or "(no transcript extracted)", language=None)

            st.markdown("## 📋 Score Report")
            show_score_card(score_report)

            st.markdown("## 🔍 Proctoring Report")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Faces",     cheating_report["avg_faces"])
            m2.metric("No-face Events",cheating_report["no_face_events"])
            m3.metric("Multi-face",    cheating_report["multiple_face_events"])
            m4.metric("Gaze Away",     cheating_report["gaze_away_events"])

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Too Close",     cheating_report.get("too_close_events", 0))
            m6.metric("Drowsy Events", cheating_report.get("drowsy_events", 0))
            m7.metric("Audio Spikes",  cheating_report.get("audio_spike_events", 0))
            m8.metric("Multi-Speaker", "Yes" if cheating_report.get("multiple_speakers") else "No")

            show_cheating_report(cheating_report)

# ── TEACHER DASHBOARD ─────────────────────────────────────
elif nav == "📊 Teacher Dashboard":
    st.title("Teacher Dashboard")

    db = st.session_state.results_db
    if not db:
        st.info("No results yet. Students must complete tests first.")
        st.stop()

    # Summary metrics
    total_students = len(db)
    cheating_count = sum(
        1 for tests in db.values()
        for data in tests.values()
        if any([
            data.get("cheating_report",{}).get("copy_paste_detected"),
            data.get("cheating_report",{}).get("ai_content_detected"),
            data.get("cheating_report",{}).get("cheating_detected"),
        ])
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students",   total_students)
    c2.metric("Cheating Flagged", cheating_count)
    c3.metric("Tests Submitted",  sum(len(t) for t in db.values()))

    st.markdown("---")

    # Export button
    if st.button("📥 Export Full Report (JSON)"):
        report_json = json.dumps(db, indent=2, default=str)
        st.download_button("⬇️ Download JSON", report_json,
                           "exam_report.json", "application/json")

    st.markdown("---")
    st.markdown("## Student Results")

    for student, tests in db.items():
        grand_scored = grand_max = 0
        any_cheating = False

        for test_type, data in tests.items():
            total = data.get("score_report", {}).get("__total__", {})
            grand_scored += total.get("scored", 0)
            grand_max    += total.get("max", 0)
            cr = data.get("cheating_report", {})
            if any([cr.get("copy_paste_detected"), cr.get("ai_content_detected"), cr.get("cheating_detected")]):
                any_cheating = True

        pct = round(grand_scored / max(grand_max, 1) * 100, 1)
        integrity_label = pill("⚠ FLAGGED", "err") if any_cheating else pill("✅ CLEAN", "ok")

        with st.expander(f"👤 {student}  —  {grand_scored}/{grand_max} ({pct}%)  {integrity_label}", expanded=False):
            tabs = st.tabs([f"📝 {tt}" for tt in tests.keys()])
            for tab, (test_type, data) in zip(tabs, tests.items()):
                with tab:
                    sr = data.get("score_report", {})
                    cr = data.get("cheating_report", {})
                    tot = sr.get("__total__", {"scored": 0, "max": 0})
                    st.markdown(f"**Score: {tot.get('scored',0)} / {tot.get('max',0)}**")
                    show_score_card(sr)
                    st.markdown("**Integrity:**")
                    show_cheating_report(cr)

    st.markdown("---")
    if st.button("🗑 Clear All Results"):
        st.session_state.results_db = {}
        st.success("All results cleared.")
        st.rerun()
