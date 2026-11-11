import io 
import pdfplumber 

def pdf_to_text(pdf_bytes: bytes, threshold:int = 50) -> str: 
    text:str = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf: 
        text = "\n".join(page.extract_text() or "" for page in pdf.pages) 
        if len(text.strip()) < threshold:
            return _ocr_pdf(pdf) 
    return text 

def _ocr_pdf(pdf: pdfplumber.PDF) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    images = (page.to_image(resolution=300).original for page in pdf.pages)
    return "\n".join(
        pytesseract.image_to_string(img, lang="sin+eng+tam")
        for img in images 
    )