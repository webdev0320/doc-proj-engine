import fitz
import json

doc = fitz.open(r"d:\projects\doc-project\engine\files\test-1_merged_compressed.pdf")
print("Total pages:", len(doc))

for i in range(len(doc)):
    page = doc[i]
    text = page.get_text("text").strip()
    first_lines = [line.strip() for line in text.split("\n") if line.strip()][:5]
    print(f"Page {i+1}: length={len(text)} | Lines: {first_lines}")
