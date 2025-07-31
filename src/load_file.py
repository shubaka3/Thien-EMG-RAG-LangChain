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

# Load environment variables
load_dotenv('./.env')

# New libraries for OCR
try:
    import fitz # PyMuPDF
    from PIL import Image
    import pytesseract
    # Configure tesseract path if needed from .env
    tesseract_cmd_path = os.getenv("TESSERACT_CMD_PATH")
    if tesseract_cmd_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
except ImportError:
    print("PyMuPDF, Pillow, or pytesseract not found. OCR functionality will be limited.")
    fitz = None
    Image = None
    pytesseract = None

_EProviders:Dict[Literal['openai', 'ollama'], Type[Embeddings]] = {
    'openai': OpenAIEmbeddings,
    'ollama': OllamaEmbeddings
}

def get_embedding_model(provider:Literal['openai', 'ollama'], model:str, **kwargs) -> Embeddings:
    """
    Retrieves the embedding model based on the provider and model name.

    Args:
        provider (Literal['openai', 'ollama']): The embedding model provider ('openai' or 'ollama').
        model (str): The name of the embedding model.
        **kwargs: Additional arguments for the embedding model.

    Returns:
        Embeddings: An instance of the embedding model.
    """
    return _EProviders[provider](model=model, **kwargs)

def _extract_text_from_pdf_with_ocr(pdf_path: str) -> List[str]:
    """
    Extracts text from PDF, using OCR if text cannot be extracted directly.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[str]: List of text strings, each corresponding to a page.
    """
    if not fitz or not Image or not pytesseract:
        print("OCR dependencies (PyMuPDF, Pillow, pytesseract) are not fully installed. Skipping OCR for PDF.")
        loader = PyPDFLoader(pdf_path)
        return [doc.page_content for doc in loader.load()] # Fallback to direct text extraction

    doc = fitz.open(pdf_path)
    text_per_page = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        if not text.strip():  # If no text or only whitespace, try OCR
            print(f"No text found on page {page_num + 1} of {os.path.basename(pdf_path)}. Attempting OCR...")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang='vie+eng') # Try both Vietnamese and English
            text_per_page.append(ocr_text)
        else:
            text_per_page.append(text)
    return text_per_page


def load_documents(path:str) -> List[Document]:
    """
    Loads documents from a given directory, including OCR support for PDFs.
    This function processes all files in the directory.

    Args:
        path (str): Path to the directory containing document files.

    Returns:
        List[Document]: List of loaded Langchain Document objects.
    """
    documents:list[Document] = []
    
    # Get chunking parameters from environment variables
    chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath): # Ensure it's a file, not a directory
            print(f"Processing file: {filename}")
            if filename.endswith('.pdf'):
                try:
                    page_contents = _extract_text_from_pdf_with_ocr(filepath)
                    # Create Langchain Documents from each page
                    for i, content in enumerate(page_contents):
                        if content.strip(): # Only add pages with content
                            docs_from_page = text_splitter.split_text(content)
                            for chunk in docs_from_page:
                                documents.append(Document(
                                    page_content=chunk, 
                                    metadata={
                                        'source': filename, 
                                        'content_type': 'application/pdf', 
                                        'page': i + 1, 
                                        'total_pages': len(page_contents)
                                    }
                                ))
                except Exception as e:
                    print(f"Error processing PDF {filename} with OCR: {e}. Skipping or falling back.")
                    # Fallback to PyPDFLoader if OCR fails
                    try:
                        loader = PyPDFLoader(filepath)
                        fdocs = [
                            Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'application/pdf', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
                            for doc in loader.load_and_split(text_splitter=text_splitter)
                        ]
                        documents.extend(fdocs)
                    except Exception as e_fallback:
                        print(f"Failed to load PDF {filename} even with fallback: {e_fallback}. Skipping.")

            elif filename.endswith('.md'):
                loader = UnstructuredMarkdownLoader(filepath)
                fdocs = [
                    Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'text/markdown', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
                    for doc in loader.load_and_split(text_splitter=text_splitter)
                ]
                documents.extend(fdocs)
            elif filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                # Split .txt file content
                docs_from_txt = text_splitter.split_text(content)
                for chunk in docs_from_txt:
                    documents.append(Document(page_content=chunk, metadata={'source': filename, 'content_type': 'text/plain', 'page': 1, 'total_pages': 1}))
            else:
                print(f"Unsupported file type: {filename}. Skipping.")
                    
    return documents

def process_single_document(filepath: str) -> List[Document]:
    """
    Loads and chunks a single document from a given file path.
    This is a helper for the new upload endpoint.

    Args:
        filepath (str): Path to the single document file.

    Returns:
        List[Document]: List of loaded Langchain Document objects from the single file.
    """
    documents: List[Document] = []
    
    # Get chunking parameters from environment variables
    chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    filename = os.path.basename(filepath)
    print(f"Processing single file: {filename}")

    if filename.endswith('.pdf'):
        try:
            page_contents = _extract_text_from_pdf_with_ocr(filepath)
            for i, content in enumerate(page_contents):
                if content.strip():
                    docs_from_page = text_splitter.split_text(content)
                    for chunk in docs_from_page:
                        documents.append(Document(
                            page_content=chunk, 
                            metadata={
                                'source': filename, 
                                'content_type': 'application/pdf', 
                                'page': i + 1, 
                                'total_pages': len(page_contents)
                            }
                        ))
        except Exception as e:
            print(f"Error processing PDF {filename} with OCR: {e}. Falling back to PyPDFLoader.")
            try:
                loader = PyPDFLoader(filepath)
                fdocs = [
                    Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'application/pdf', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
                    for doc in loader.load_and_split(text_splitter=text_splitter)
                ]
                documents.extend(fdocs)
            except Exception as e_fallback:
                print(f"Failed to load PDF {filename} even with fallback: {e_fallback}. Skipping.")

    elif filename.endswith('.md'):
        loader = UnstructuredMarkdownLoader(filepath)
        fdocs = [
            Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'text/markdown', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
            for doc in loader.load_and_split(text_splitter=text_splitter)
        ]
        documents.extend(fdocs)
    elif filename.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        docs_from_txt = text_splitter.split_text(content)
        for chunk in docs_from_txt:
            documents.append(Document(page_content=chunk, metadata={'source': filename, 'content_type': 'text/plain', 'page': 1, 'total_pages': 1}))
    else:
        print(f"Unsupported file type: {filename}. Skipping.")
            
    return documents

# Main function just for testing document loading
def main():
    load_dotenv('./.env')
    # Example of how to load documents
    storage_path = os.getenv("SERVER_STORAGE_PATH", './storage')
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
        print(f"Created storage directory: {storage_path}")
    
    documents = load_documents(storage_path)
    print(f"Loaded {len(documents)} documents.")
    for doc in documents[:5]: # Print first 5 documents for verification
        print(f"--- Document ---")
        print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()

