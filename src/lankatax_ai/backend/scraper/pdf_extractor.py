import io 
import pdfplumber 

PAGE_MARKER = "<!--PAGE:{n}-->"

def safe_str(s:str|None) -> str:
    return "" if s is None else s 

def pdf_to_text(pdf_bytes: bytes, threshold:int = 50) -> str:
    text:str = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf: 
        text = ""
        for i,page in enumerate(pdf.pages,start=1):
            page_text = PAGE_MARKER.format(n=i) 
            page_text += safe_str(page.extract_text())
            text += page_text
        if len(text.strip()) < threshold:
            return _ocr_pdf(pdf) 
    return text 

def _ocr_pdf(pdf: pdfplumber.PDF) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    images = (page.to_image(resolution=300).original for page in pdf.pages)
    text = "" 
    for i, img in enumerate(images,start=1):
        page_text = PAGE_MARKER.format(n=i)
        page_text += safe_str(pytesseract.image_to_string(img, lang="sin+eng+tam"))
        text += page_text
    return text
    


