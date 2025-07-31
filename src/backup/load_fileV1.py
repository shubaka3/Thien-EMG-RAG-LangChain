#load_file.py
from collections import defaultdict
import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from typing import Literal, Dict, Type
from pymilvus import connections, Collection
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter
# new
from langchain.schema import Document as LangchainDocument
from langchain.vectorstores import Milvus as LangchainMilvus

_EProviders:Dict[Literal['openai', 'ollama'], Type[Embeddings]] = {
    'openai': OpenAIEmbeddings,
    'ollama': OllamaEmbeddings
}

def get_embedding_model(provider:Literal['openai', 'ollama'], model:str, **kwargs) -> Embeddings:
    return _EProviders[provider](model=model, **kwargs)

def load_documents(path:str):
    documents:list[Document] = []
    for filename in os.listdir(path):
        print(f"Processing file: {filename}")
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(path, filename))
            fdocs = [
                Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'application/pdf', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
                for doc in loader.load_and_split(text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
            ]
            documents.extend(fdocs)
            # documents.extend(loader.load())
        elif filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(os.path.join(path, filename))
            fdocs = [
                Document(page_content=doc.page_content, metadata={'source': filename, 'content_type': 'text/markdown', 'page': doc.metadata.get('page', 1), 'total_pages': doc.metadata.get('total_pages', 1)})
                for doc in loader.load_and_split(text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
            ]
            documents.extend(fdocs)
        elif filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
            documents.append(Document(page_content=content, metadata={'source': filename, 'content_type': 'text/plain', 'page': 1, 'total_pages': 1}))
        else:
            print(f"Unsupported file type: {filename}. Skipping.")
                
    return documents

def add_documents_to_db(
    uri:str,
    collection_name:str,
    documents:list[Document], 
    embedding_provider:Literal['openai', 'ollama'] = 'openai', 
    model:str = 'text-embedding-3-large'
) -> Milvus:
    embeddings = get_embedding_model(embedding_provider, model)
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name,
        drop_old=True
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    
    print(f"Adding {len(documents)} documents to Milvus collection '{collection_name}'...")
    vectorstore.add_documents(documents, ids=uuids)
    print(f"Documents added successfully to collection '{collection_name}'")
    
    return vectorstore

def list_sources_in_db(uri:str, collection_name:str) -> dict[str, int]:
    connections.connect(uri=uri)
    collection = Collection(name=collection_name)
    
    print(f"Listing documents in collection '{collection_name}'...")
    documents = collection.query("pk != ''", output_fields=["*"])
    # group all source documents by their metadata source
    source_count = defaultdict(int)
    for doc in documents:
        source = doc.get('source', 'unknown')
        source_count[source] += 1
    return dict(source_count)

def delete_source_from_db(uri:str, collection_name:str, source:str) -> bool:
    connections.connect(uri=uri)
    collection = Collection(name=collection_name)
    
    print(f"Deleting documents from collection '{collection_name}' with source '{source}'...")
    result = collection.delete(f"source == '{source}'")
    print(f"Delete result: {result}")
    collection.flush() 

def get_documents_from_db(uri: str, collection_name: str, limit: int = 100) -> list[Document]:
    embeddings = get_embedding_model(
        provider=os.getenv("SERVER_EMBEDDING_PROVIDER", "openai"),
        model=os.getenv("SERVER_EMBEDDING_MODEL", "text-embedding-3-large")
    )

    vectorstore = LangchainMilvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name
    )

    # Lấy tất cả document có sẵn bằng retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": limit})
    docs = retriever.get_relevant_documents("dummy text")  # fake query để trả document

    return docs



def main():
    add_documents_to_db(
        uri='http://localhost:19530',
        collection_name='test_data2',
        documents=load_documents('./storage'),
        embedding_provider='openai',
        model='text-embedding-3-large'
    )

    # List sources in the database
    sources = list_sources_in_db('http://localhost:19530', 'test_data2')
    print("Sources in the database:", sources)

    # Delete a specific source from the database
    # delete_source_from_db('http://localhost:19530', 'test_data2', 'sound.md')
    # Verify the deletion
    sources_after_deletion = list_sources_in_db('http://localhost:19530', 'test_data2')
    print("Sources after deletion:", sources_after_deletion)

if __name__ == "__main__":
    load_dotenv('./.env')
    main()