import os
# STABILITY FLAGS
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_ONEDNN'] = 'OFF'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_new_ir'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import traceback

PDF_PATH = r"F:\AAFS\My Documents\Fatma\coding projects\Random proj infos\training\test.pdf"
JSON_OUTPUT = "extracted_texts.json"
NUM_CLUSTERS = 10

def extract_text_from_pdf(pdf_path, json_path):
    """
    Extracts text from each page. Uses embedded text if available (fast),
    otherwise falls back to PaddleOCR (slower).
    """
    if os.path.exists(json_path):
        print(f"Loading previously extracted text from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"Extracting text from {pdf_path}. This may take a while for 600 pages...")
    extracted_data = []
    
    try:
        doc = fitz.open(pdf_path)
        ocr = None
        
        for i in range(len(doc)):
            page = doc[i]
            
            # Fast Path: Try embedded text first
            text = page.get_text("text").strip()
            
            # If the page has very little embedded text, it's likely a scanned image. Use OCR.
            # Use embedded text only to bypass the PaddleOCR Windows bug
            if len(text) < 50:
                print(f"  Page {i+1}/{len(doc)}: No embedded text found, skipping page for clustering.")
                text = ""
            else:
                print(f"  Page {i+1}/{len(doc)}: Extracted embedded text instantly.")
                
            extracted_data.append({
                "page_num": i + 1,
                "text": text
            })
            
            # Save incrementally in case it crashes
            if (i + 1) % 10 == 0:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=4)
                    
        doc.close()
        
        # Final save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4)
            
        print("Extraction complete!")
        return extracted_data
        
    except Exception as e:
        print("[Error] An error occurred during extraction:")
        traceback.print_exc()
        return []

def run_clustering(extracted_data, num_clusters=10):
    print(f"\n--- Running KMeans Clustering ({num_clusters} clusters) ---")
    
    # Filter out empty pages
    valid_docs = [item for item in extracted_data if len(item['text'].strip()) > 10]
    texts = [item['text'] for item in valid_docs]
    
    if not texts:
        print("No valid text found to cluster.")
        return
        
    # 1. TF-IDF Vectorization (Finds important words, ignores stop words)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.8,   # Ignore words that appear in >80% of documents (too common)
        min_df=2,     # Ignore words that appear in <2 documents (too rare)
        ngram_range=(1, 2) # Look at single words and pairs of words (e.g. "bank statement")
    )
    
    X = vectorizer.fit_transform(texts)
    
    # 2. KMeans Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    # 3. Analyze Results
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    clusters_info = {}
    
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        
        # Find which pages belong to this cluster
        page_nums = []
        for doc_idx, label in enumerate(kmeans.labels_):
            if label == i:
                page_nums.append(valid_docs[doc_idx]['page_num'])
                
        clusters_info[i] = {
            "size": len(page_nums),
            "top_words": top_words,
            "sample_pages": page_nums[:5] # Show first 5 pages as examples
        }
        
    # 4. Print Report
    print("\n" + "="*50)
    print("CLUSTERING REPORT (Please Review & Label)")
    print("="*50)
    for cluster_id, info in clusters_info.items():
        print(f"\nCluster {cluster_id} (Contains {info['size']} pages):")
        print(f"Top Words : {', '.join(info['top_words'])}")
        print(f"Sample Pgs: {', '.join(map(str, info['sample_pages']))}")
        print("Proposed Name: _________________")

if __name__ == "__main__":
    print("Starting Auto-Cluster Process...")
    data = extract_text_from_pdf(PDF_PATH, JSON_OUTPUT)
    if data:
        # You can tweak the number of clusters if you think there are more/fewer document types
        run_clustering(data, NUM_CLUSTERS)
