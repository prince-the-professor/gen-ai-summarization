from pdf2image import convert_from_path
import pytesseract
import os

def extract_text_from_pdf(pdf_path, max_pages=10):
    print("🔁 Converting PDF pages to images...")
    images = convert_from_path(pdf_path)

    if max_pages:
        images = images[:max_pages]

    all_text = ""
    for i, img in enumerate(images):
        print(f"🔍 Running OCR on page {i+1}/{len(images)}...")
        text = pytesseract.image_to_string(img)
        all_text += text + "\n"

    print("✅ OCR extraction complete.")
    return all_text
