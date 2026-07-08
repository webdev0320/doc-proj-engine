import os
import sys

# Override BACKEND_URL to point to local backend
os.environ['BACKEND_URL'] = 'http://localhost:3001'

from main import run_pipeline

blob_id = "cmqcmtkg50003chpratdyq4o7"
pdf_path = os.path.join(os.path.dirname(__file__), "../storage/blobs/17a78124-6c9a-4044-bfd3-5adf500fbc93-46d1afaf-a91d-4a40-93d8-11b5fe26e4f8-3740584191741.pdf")

print(f"Running pipeline for blob {blob_id} with pdf {pdf_path}")
run_pipeline(blob_id, pdf_path)
print("Pipeline run finished.")
