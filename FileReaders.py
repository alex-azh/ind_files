import docx # pip install python-docx
import fitz # pip install PyMuPDF


def DOCReader(file: str):
    try:
        doc = docx.Document(file)
        text = ""
        for p in doc.paragraphs:
            text += p.text
        return text
    except:
        return None


def PDFReader(file):
    with fitz.open(file) as pdf_file:
        # Получаем количество страниц в документе
        num_pages = pdf_file.page_count
        s=""
        # Читаем содержимое каждой страницы и выводим его в консоль
        for page_num in range(num_pages):
            page = pdf_file[page_num]
            s += page.get_text()
    return s

