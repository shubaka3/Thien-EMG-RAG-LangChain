# src/load_file.py
import os
import uuid
import logging
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter

# Import các thành phần từ các file khác của dự án
from src.milvus_langchain import milvus_service
from src.database import AIModel, Collection

# Tải biến môi trường
load_dotenv('./.env')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thư viện cho OCR
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    # Cấu hình đường dẫn tesseract nếu cần từ .env
    tesseract_cmd_path = os.getenv("TESSERACT_CMD_PATH")
    if tesseract_cmd_path and os.path.exists(tesseract_cmd_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
except ImportError:
    logging.warning("Không tìm thấy PyMuPDF, Pillow hoặc pytesseract. Chức năng OCR sẽ bị hạn chế.")
    fitz = None
    Image = None
    pytesseract = None

def _extract_text_from_pdf_with_ocr(pdf_path: str) -> List[Document]:
    """
    Trích xuất văn bản từ PDF, sử dụng OCR cho các trang chỉ chứa hình ảnh.
    Mỗi trang của PDF sẽ trở thành một đối tượng Document của LangChain.
    """
    if not fitz:
        logging.warning("PyMuPDF (fitz) chưa được cài đặt. Chỉ sử dụng PyPDFLoader.")
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    doc = fitz.open(pdf_path)
    documents = []
    filename = os.path.basename(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        page_content = ""
        
        # Nếu không có văn bản hoặc văn bản quá ngắn, thử OCR
        if not text or len(text.strip()) < 20:
            if pytesseract and Image:
                logging.info(f"Trang {page_num + 1} của {filename} có ít văn bản, đang thử OCR...")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    # Thử OCR với cả tiếng Việt và tiếng Anh
                    ocr_text = pytesseract.image_to_string(img, lang='vie+eng')
                    page_content = text + "\n" + ocr_text # Kết hợp cả hai nếu có
                except Exception as ocr_error:
                    logging.error(f"Lỗi OCR trên trang {page_num + 1}: {ocr_error}")
                    page_content = text # Quay lại dùng text gốc nếu OCR lỗi
            else:
                page_content = text
        else:
            page_content = text

        documents.append(Document(
            page_content=page_content,
            metadata={
                'source': filename,
                'page': page_num + 1,
                'total_pages': len(doc)
            }
        ))
    return documents

def _load_and_split_documents(file_path: str) -> List[Document]:
    """
    Tải và chia nhỏ một tài liệu duy nhất dựa trên định dạng của nó.
    """
    filename = os.path.basename(file_path).split('_', 1)[1] # Lấy tên file gốc
    file_extension = os.path.splitext(filename)[1].lower()
    
    docs_before_split = []

    if file_extension == '.pdf':
        logging.info(f"Đang xử lý tệp PDF: {filename} với OCR (nếu cần).")
        docs_before_split = _extract_text_from_pdf_with_ocr(file_path)
    elif file_extension == '.md':
        logging.info(f"Đang xử lý tệp Markdown: {filename}")
        headers_to_split_on = [("#", "Header1"), ("##", "Header2"), ("###", "Header3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        docs_before_split = markdown_splitter.split_text(markdown_text)
        # Gán metadata cho file markdown
        for doc in docs_before_split:
            doc.metadata['source'] = filename
    elif file_extension == '.json':
        logging.info(f"Đang xử lý tệp JSON: {filename}")
        # Giả định mỗi object trong array là một document
        loader = JSONLoader(file_path, jq_schema=".[]", text_content=False)
        docs_before_split = loader.load()
    else:
        logging.warning(f"Bỏ qua loại tệp không được hỗ trợ: {filename}")
        return []

    if not docs_before_split:
        return []

    # Chia nhỏ các document đã tải
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    
    final_chunks = text_splitter.split_documents(docs_before_split)
    
    # Đảm bảo metadata 'source' được giữ lại
    for chunk in final_chunks:
        if 'source' not in chunk.metadata:
             chunk.metadata['source'] = filename
    
    logging.info(f"Đã chia tài liệu {filename} thành {len(final_chunks)} chunks.")
    return final_chunks


def process_and_embed_document(file_path: str, collection_info: Collection, ai_info: AIModel) -> bool:
    """
    Hàm chính: Tải, chia nhỏ, tạo embedding và thêm tài liệu vào Milvus.
    """
    try:
        # 1. Tải và chia nhỏ tài liệu
        chunks = _load_and_split_documents(file_path)

        if not chunks:
            logging.warning(f"Không có chunk nào được tạo từ tài liệu. Bỏ qua.")
            return True

        # 2. Thêm các chunk vào Milvus (đã bao gồm việc tạo embedding)
        logging.info(f"Bắt đầu thêm {len(chunks)} chunks vào Milvus collection '{collection_info.milvus_collection_name}'...")
        pks = milvus_service.add_documents(
            collection_name=collection_info.milvus_collection_name,
            documents=chunks,
            embedding_model_provider=ai_info.provider,
            embedding_model_name=ai_info.embedding_model_name,
            api_key=ai_info.api_key,
            embedding_dim=ai_info.embedding_dim
        )

        if pks:
            logging.info(f"Đã thêm thành công {len(pks)} chunks vào Milvus.")
            return True
        else:
            logging.error(f"Không thể thêm chunks vào Milvus.")
            return False

    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng trong quá trình xử lý và nhúng tài liệu: {e}", exc_info=True)
        return False
