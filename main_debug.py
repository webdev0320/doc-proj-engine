import os
# STABILITY FLAGS
# os.environ['FLAGS_use_mkldnn'] = '0'
# os.environ['PADDLE_ONEDNN'] = 'OFF'
# os.environ['FLAGS_enable_new_executor'] = '0'
# os.environ['FLAGS_enable_new_ir'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import requests
import cv2
import numpy as np
import json
import fitz
import paramiko
import boto3
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
# import easyocr
from paddleOCR import classify_page
from crypto_utils import decrypt_file
from dotenv import load_dotenv

load_dotenv()

# Initialize EasyOCR once at startup
print("Skipping EasyOCR model loading for debug...")
# reader = easyocr.Reader(['en'], gpu=False)
reader = None
print("Mock Reader ready.")

app = FastAPI(title="IDP Engine - PaddleOCR DEBUG")

@app.get("/")
def root():
    return {"status": "running", "mode": "debug-no-ocr"}

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
