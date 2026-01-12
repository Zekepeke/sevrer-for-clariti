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

# Assuming preprocessed image is in bytes and preprocess_face_from_bytes
# and embed_face in current expo app
def download_image_from_supabase(bucket: str, path: str) -> bytes:
    # supabase-py returns raw bytes for download
    data = supabase.storage.from_(bucket).download(path)
    return data


# FastAPI app initialization
app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
)

# Initialize FaceNet model
mp_face = mp.solutions.face_detection
facenet = FaceNet() # 512-dimensional embeddings

