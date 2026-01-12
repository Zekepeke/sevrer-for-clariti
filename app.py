import os
import datetime
from typing import Literal, Optional, List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client


# Initialize Supabase client

# Load Supabase credentials from environment variables
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)



# FastAPI app initialization
app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
)

# Initialize FaceNet model
mp_face = mp.solutions.face_detection
embedder = FaceNet()  # 512-D embeddings


class ProcessRequest(BaseModel):
    bucket: str                   # Supabase storage bucket
    path: str                     # Supabase object path
    memory_id: str                # UUID of the row in `memories`
    mode: Literal["enroll", "match"]
    # Maybe use user_id here later if we want to scope matches by user
    # user_id: Optional[str] = None
    
# Assuming preprocessed image is in bytes and preprocess_face_from_bytes
# and embed_face in current expo app
def download_image_from_supabase(bucket: str, path: str) -> bytes:
    try:
        data = supabase.storage.from_(bucket).download(path)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Supabase download failed: {e}")

def preprocess_face_from_bytes(img_bytes: bytes) -> tuple[np.ndarray, Dict[str, Any], Optional[float]]:
    """
    Decode image bytes, run MediaPipe face detection, return:
    - cropped, normalized face image (160x160 RGB float32)
    - bbox dict (pixel coords)
    - confidence score
    """
    
    # Decode to OpenCV image
    file_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w, _ = img.shape

    # Run MediaPipe face detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            raise ValueError("No face detected")

        # Take the first detection
        det = results.detections[0]
        bbox_rel = det.location_data.relative_bounding_box

        x_min = max(int(bbox_rel.xmin * w), 0)
        y_min = max(int(bbox_rel.ymin * h), 0)
        x_max = min(int((bbox_rel.xmin + bbox_rel.width) * w), w)
        y_max = min(int((bbox_rel.ymin + bbox_rel.height) * h), h)

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid face bounding box")

        face_img = img[y_min:y_max, x_min:x_max]

        bbox = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }

        confidence = float(det.score[0]) if det.score else None

    # Resize to FaceNet input size (typically 160x160)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32")
    face_img = (face_img - 127.5) / 128.0
    

    return face_img, bbox, confidence


def embed_face(face_img: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(batch)[0]
    return embedding.astype("float32")


def cosine_similarity(a, b) -> float:
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def enroll_embedding(
    memory_id: str,
    emb_list: List[float],
    face_index: int = 0,
    bbox: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
):
    """
    Insert a row into memory_faces for this memory.
    For now we only handle a single face -> face_index = 0.
    """
    try:
        res = supabase.table("memory_faces").insert(
            {
                "memory_id": memory_id,
                "profile_id": None,   # you can fill this later when a face is labeled
                "face_index": face_index,
                "embedding": emb_list,   # works with vector(512) extension
                "bbox": bbox,
                "confidence": confidence,
            }
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    inserted = res.data[0] if res.data else None
    return {
        "ok": True,
        "mode": "enroll",
        "saved": True,
        "id": inserted["id"] if inserted else None,
    }


def match_embedding(
    emb_list: List[float],
):
    """
    Compare the probe embedding against all rows in memory_faces
    and return the best match. (You can later scope by user/group.)
    """
    try:
        res = supabase.table("memory_faces").select(
            "id,memory_id,profile_id,embedding"
        ).execute()
        rows = res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB read failed: {e}")

    if not rows:
        return {
            "ok": True,
            "mode": "match",
            "match": None,
            "reason": "no embeddings in DB",
        }

    best_row = None
    best_sim = -1.0

    for row in rows:
        cand_emb = row["embedding"]
        sim = cosine_similarity(emb_list, cand_emb)
        if sim > best_sim:
            best_sim = sim
            best_row = row

    threshold = 0.6  # tune this!
    if best_sim < threshold or best_row is None:
        match = None
    else:
        match = {
            "id": best_row["id"],
            "memory_id": best_row["memory_id"],
            "profile_id": best_row.get("profile_id"),
            "similarity": best_sim,
        }

    return {
        "ok": True,
        "mode": "match",
        "match": match,
        "best_similarity": best_sim,
        "threshold": threshold,
    }


@app.post("/process")
def process_image(payload: ProcessRequest):
    # 1. Download image from Supabase
    img_bytes = download_image_from_supabase(payload.bucket, payload.path)

    # 2. Detect + crop face, then embed
    try:
        face_img, bbox, confidence = preprocess_face_from_bytes(img_bytes)
        embedding = embed_face(face_img)
        emb_list = embedding.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face processing error: {e}")

    # 3. Enroll or match
    if payload.mode == "enroll":
        return enroll_embedding(
            memory_id=payload.memory_id,
            emb_list=emb_list,
            face_index=0,
            bbox=bbox,
            confidence=confidence,
        )
    else:
        return match_embedding(
            emb_list=emb_list,
        )


@app.get("/health")
def health():
    return {
        "ok": True,
        "time": datetime.datetime.utcnow().isoformat(),
        "status": "healthy",
    }