#load_file.py
from collections import defaultdict
import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from typing import Literal, Dict, Type, List
from pymilvus import connections, Collection
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter

# Tải biến môi trường
load_dotenv('./.env')

# Thư viện mới cho OCR
try:
    import fitz # PyMuPDF
    from PIL import Image
    import pytesseract
    # Cấu hình đường dẫn tesseract nếu cần từ .env
    tesseract_cmd_path = os.getenv("TESSERACT_CMD_PATH")
    if tesseract_cmd_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
except ImportError:
    print("Không tìm thấy PyMuPDF, Pillow hoặc pytesseract. Chức năng OCR sẽ bị hạn chế.")
    fitz = None
    Image = None
    pytesseract = None

_EProviders:Dict[Literal['openai', 'ollama'], Type[Embeddings]] = {
    'openai': OpenAIEmbeddings,
    'ollama': OllamaEmbeddings
}

def get_embedding_model(provider:Literal['openai', 'ollama'], model:str, **kwargs) -> Embeddings:
    """
    Truy xuất mô hình nhúng dựa trên nhà cung cấp và tên mô hình.

    Args:
        provider (Literal['openai', 'ollama']): Nhà cung cấp mô hình nhúng ('openai' hoặc 'ollama').
        model (str): Tên của mô hình nhúng.
        **kwargs: Các đối số bổ sung cho mô hình nhúng.

    Returns:
        Embeddings: Một thể hiện của mô hình nhúng.
    """
    return _EProviders[provider](model=model, **kwargs)

def _extract_text_from_pdf_with_ocr(pdf_path: str) -> List[str]:
    # [feauture] sửa chỗ này, pdf có 2 dạng pdf thuần và ảnh, phải có extract text từ pdf thuần, nếu không có thì mới dùng OCR
    #  thêm logic check khoảng trống trong pdf để xem có ảnh trong đó không hoặc tài liệu quá ít so với số lượng trung bình
    # sau đó dùng ocr cho các page đấy và cũng trả về, ocr thì phải ocr theo tọa độ để trả về đúng box của data
    #  ưu tiên lưu full box hoặc box in box để giữ context
    # Nên làm 1 API riêng để xử lý OCR, có thể dùng FastAPI hoặc Flask
    """
    Trích xuất văn bản từ PDF, sử dụng OCR nếu không thể trích xuất văn bản trực tiếp.

    Args:
        pdf_path (str): Đường dẫn đến tệp PDF.

    Returns:
        List[str]: Danh sách các chuỗi văn bản, mỗi chuỗi tương ứng với một trang.
    """
    if not fitz or not Image or not pytesseract:
        print("Các phụ thuộc OCR (PyMuPDF, Pillow, pytesseract) chưa được cài đặt đầy đủ. Bỏ qua OCR cho PDF.")
        loader = PyPDFLoader(pdf_path)
        return [doc.page_content for doc in loader.load()] # Quay lại trích xuất văn bản trực tiếp

    doc = fitz.open(pdf_path)
    text_per_page = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        if not text.strip():  # Nếu không có văn bản hoặc chỉ có khoảng trắng, thử OCR
            print(f"Không tìm thấy văn bản trên trang {page_num + 1} của {os.path.basename(pdf_path)}. Đang thử OCR...")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang='vie+eng') # Thử cả tiếng Việt và tiếng Anh
            text_per_page.append(ocr_text)
        else:
            text_per_page.append(text)
    return text_per_page

def process_single_document(file_path: str) -> List[Document]:
    """
    Tải và xử lý một tài liệu duy nhất, thêm PK duy nhất và các metadata khác.
    """
    documents: List[Document] = []
    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()
    # [feauture] nên thêm hỗ trợ docx,image,txt,audio
    if file_extension == '.pdf':
        print(f"Đang xử lý tệp PDF: {filename}")
        pages_content = _extract_text_from_pdf_with_ocr(file_path)
        total_pages = len(pages_content)
        for i, page_content in enumerate(pages_content):
            # Chia nội dung PDF theo ký tự để đảm bảo các chunk dễ quản lý
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(page_content)
            for chunk_idx, chunk in enumerate(chunks):
                doc_pk = str(uuid4()) # Tạo PK duy nhất cho mỗi chunk
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            'pk': doc_pk, # Khóa chính được tạo ở đây
                            'source': filename,
                            'content_type': 'application/pdf',
                            'page': i + 1,
                            'total_pages': total_pages
                        }
                    )
                )
    elif file_extension == '.md':
        print(f"Đang xử lý tệp Markdown: {filename}")
        # Sử dụng MarkdownHeaderTextSplitter cho markdown có cấu trúc
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        md_docs = markdown_splitter.split_text(markdown_text)
        for doc in md_docs:
            doc_pk = str(uuid4()) # Tạo PK duy nhất cho mỗi chunk
            # Đảm bảo metadata từ splitter được giữ lại, thêm PK
            doc.metadata['pk'] = doc_pk # Khóa chính được tạo ở đây
            doc.metadata['source'] = filename
            doc.metadata['content_type'] = 'text/markdown'
            documents.append(doc)

    elif file_extension == '.json':
        print(f"Đang xử lý tệp JSON: {filename}")
        try:
            loader = JSONLoader(file_path, jq_schema=".[]") # Điều chỉnh schema theo cấu trúc JSON của bạn
            json_docs = loader.load_and_split(text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
            for doc in json_docs:
                doc_pk = str(uuid4()) # Tạo PK duy nhất cho mỗi chunk
                doc.metadata['pk'] = doc_pk # Khóa chính được tạo ở đây
                doc.metadata['source'] = filename
                doc.metadata['content_type'] = 'application/json'
                documents.append(doc)
        except Exception as e:
            print(f"Lỗi khi tải tệp JSON {filename}: {e}")
    else:
        print(f"Bỏ qua loại tệp không được hỗ trợ: {filename}")

    return documents

def move_file(source_path: str, destination_path: str):
    """
    Di chuyển một tệp từ source_path đến destination_path.
    Tạo thư mục đích nếu nó không tồn tại.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    try:
        os.rename(source_path, destination_path)
        print(f"Đã di chuyển tệp từ '{source_path}' đến '{destination_path}'")
    except OSError as e:
        print(f"Lỗi khi di chuyển tệp '{source_path}' đến '{destination_path}': {e}")
