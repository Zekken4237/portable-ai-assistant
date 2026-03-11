from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

images = convert_from_path(
    r"docs\Machiavelli,+The+Prince.pdf",
    poppler_path=r"C:\Users\PC\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
    dpi=300,
)

text = pytesseract.image_to_string(
    images[0],
    lang="eng",
    config="--oem 3 --psm 6",
)

print(text[:1000])
