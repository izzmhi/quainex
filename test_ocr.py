import pytesseract

try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Success! Tesseract version {version} is installed and accessible.")
    print("Your local environment is ready for the next step.")
except pytesseract.TesseractNotFoundError:
    print("❌ Error: Tesseract is not installed correctly or it's not in your system's PATH.")
    print("Please double-check your Tesseract installation.")