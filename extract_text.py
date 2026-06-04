import fitz  # PyMuPDF
import os

pdf_path = "D:/projects/doc-project/engine/files/test-1_merged_compressed.pdf"
doc = fitz.open(pdf_path)

for i in range(len(doc)):
    page = doc[i]
    text = page.get_text()
    print(f"--- PAGE {i+1} ---")
    print(text)
    print("\n\n")
doc.close()
