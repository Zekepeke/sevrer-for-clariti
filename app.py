import os
import datetime
from typing import Literal, Optional, List

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


def preprocess_face_from_bytes(img_bytes: bytes) -> np.ndarray:
    
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

    # Run MediaPipe face detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            raise ValueError("No face detected")

        # Take the first detection
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box

        h, w, _ = img.shape
        x_min = max(int(bbox.xmin * w), 0)
        y_min = max(int(bbox.ymin * h), 0)
        x_max = min(int((bbox.xmin + bbox.width) * w), w)
        y_max = min(int((bbox.ymin + bbox.height) * h), h)

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid face bounding box")

        face_img = img[y_min:y_max, x_min:x_max]

    # Resize to FaceNet input size (typically 160x160)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32")
    # Simple normalization (you can adjust to match your current code)
    face_img = (face_img - 127.5) / 128.0

    return face_img

def embed_face(face_img: np.ndarray) -> np.ndarray:
    # face_img shape: (160, 160, 3)
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
    user_id: Optional[str],
    bucket: str,
    path: str,
    emb_list: List[float],
    label: Optional[str],
):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required for enroll")

    try:
        res = supabase.table("memory_faces").insert(
            {
                "user_id": user_id,
                "bucket": bucket,
                "path": path,
                "embedding": emb_list,  # JSONB
                "label": label,
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
    user_id: Optional[str],
    emb_list: List[float],
):
    # Decide scope: match within same user_id, or global?
    query = supabase.table("memory_faces").select(
        "id,user_id,label,embedding"
    )
    # Example: only compare within this user
    if user_id:
        query = query.eq("user_id", user_id)

    try:
        res = query.execute()
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
            "user_id": best_row["user_id"],
            "label": best_row.get("label"),
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
        face_img = preprocess_face_from_bytes(img_bytes)
        embedding = embed_face(face_img)
        emb_list = embedding.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face processing error: {e}")

    # 3. Enroll or match
    if payload.mode == "enroll":
        return enroll_embedding(
            user_id=payload.user_id,
            bucket=payload.bucket,
            path=payload.path,
            emb_list=emb_list,
            label=payload.label,
        )
    else:
        return match_embedding(
            user_id=payload.user_id,
            emb_list=emb_list,
        )


@app.get("/health")
def health():
    return {
        "ok": True,
        "time": datetime.datetime.utcnow().isoformat(),
        "status": "healthy",
    }