# src/milvus_langchain.py
import os
from langchain_community.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from pymilvus import Collection as MilvusCollection, utility
from pymilvus import connections, FieldSchema, CollectionSchema, DataType
import logging
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
# THÊM IMPORT CHO GEMINI EMBEDDINGS NẾU DÙNG (CẦN CÀI langchain-google-genai)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None
    logging.warning("langchain_google_genai not installed. Gemini embeddings will not be available.")

# Load environment variables (đảm bảo file .env được tải)
from dotenv import load_dotenv
load_dotenv('./.env')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lấy thông tin kết nối Milvus từ biến môi trường
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URL = f"{MILVUS_HOST}:{MILVUS_PORT}"

# Connect to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
logging.info(f"Connected to Milvus at {MILVUS_URL}")

class MilvusService:
    def __init__(self):
        pass

    def _get_embedding_function(self, provider: str, api_key: str, embedding_model_name: str, embedding_dim: int) -> Embeddings:
        """
        Khởi tạo và trả về hàm embedding dựa trên nhà cung cấp và tên mô hình embedding cụ thể.
        'embedding_model_name' ở đây phải là tên của mô hình EMBEDDING.
        """
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider.")
            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAIEmbeddings(model=embedding_model_name, dimensions=embedding_dim)
        elif provider == "gemini":
            if not GoogleGenerativeAIEmbeddings:
                raise ImportError("langchain_google_genai not installed. Cannot use Gemini embeddings.")
            if not api_key:
                raise ValueError("Google API key is required for Gemini provider.")
            os.environ["GOOGLE_API_KEY"] = api_key
            logging.warning("Gemini embeddings might not strictly adhere to 'embedding_dim' parameter directly via Langchain's GoogleGenerativeAIEmbeddings. Check model documentation for actual dimension.")
            return GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        elif provider == "custom":
            logging.warning("Custom embedding not yet implemented. Please replace this with your custom embedding logic.")
            raise NotImplementedError("Custom embedding not yet implemented. Implement your custom embedding logic here.")
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


    def create_collection(self, collection_name: str, embedding_dim: int):
        if utility.has_collection(collection_name):
            logging.info(f"Collection '{collection_name}' already exists.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description=f"Collection for {collection_name}")
        MilvusCollection(collection_name, schema)

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection = MilvusCollection(collection_name)
        collection.create_index("vector", index_params)
        logging.info(f"Collection '{collection_name}' created with index.")

    def drop_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logging.info(f"Collection '{collection_name}' dropped successfully.")
        else:
            logging.warning(f"Collection '{collection_name}' does not exist, skipping drop.")

    def add_documents(self, collection_name: str, documents: list[Document],
                      embedding_model_provider: str, embedding_model_name: str, # embedding_model_name ở đây là tên của embedding model
                      api_key: str, embedding_dim: int):
        try:
            # Dùng embedding_model_name được truyền vào cho hàm embedding
            embeddings_func = self._get_embedding_function(
                embedding_model_provider, api_key, embedding_model_name, embedding_dim
            )

            if not utility.has_collection(collection_name):
                self.create_collection(collection_name, embedding_dim)

            collection = MilvusCollection(collection_name)
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents]

            vectors = embeddings_func.embed_documents(texts)

            entities = [
                vectors,
                texts,
                metadatas
            ]

            insert_result = collection.insert(entities)
            pks = insert_result.primary_keys
            collection.flush()
            
            logging.info(f"Added {len(pks)} documents to Milvus collection '{collection_name}'.")
            return pks
        except Exception as e:
            logging.error(f"Error adding documents to Milvus collection '{collection_name}': {e}", exc_info=True)
            return []

    def get_document_count_by_source(self, collection_name: str, source_filename: str) -> int:
        if not utility.has_collection(collection_name):
            return 0
        
        collection = MilvusCollection(collection_name)
        collection.load()
        expr = f'metadata["source"] == "{source_filename}"' 
        try:
            results = collection.query(expr=expr, output_fields=["id"])
            collection.release()
            return len(results)
        except Exception as e:
            logging.error(f"Error querying Milvus for document count by source in '{collection_name}': {e}", exc_info=True)
            collection.release()
            return 0

    def delete_documents_by_source(self, collection_name: str, source_filename: str) -> int:
        if not utility.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' does not exist, cannot delete documents by source.")
            return 0

        collection = MilvusCollection(collection_name)
        collection.load()

        expr = f'metadata["source"] == "{source_filename}"'
        
        try:
            results = collection.query(expr=expr, output_fields=["id"])
            pks_to_delete = [int(res['id']) for res in results]

            if pks_to_delete:
                ids_str = ', '.join(str(pk) for pk in pks_to_delete)
                expr = f"id in [{ids_str}]"
                logging.info(f"Delete expr: {expr}")
                collection.delete(expr)
                collection.flush()
                logging.info(f"Deleted {len(pks_to_delete)} documents")
                collection.release()
                return len(pks_to_delete)
            else:
                logging.info(f"No docs found to delete")
                collection.release()
                return 0
        except Exception as e:
            logging.error(f"Error deleting documents by source in '{collection_name}': {e}", exc_info=True)
            collection.release()
            return 0

    def search_documents(self, collection_name: str, query: str, k: int,
                         embedding_model_provider: str, embedding_model_name: str, # embedding_model_name ở đây là tên của embedding model
                         api_key: str, embedding_dim: int) -> list[Document]:
        """
        Tìm kiếm tài liệu trong một Milvus collection sử dụng embedding function được tạo động.
        """
        if not utility.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' does not exist for searching.")
            return []

        try:
            # Dùng embedding_model_name được truyền vào cho hàm embedding
            embeddings = self._get_embedding_function(
                embedding_model_provider, api_key, embedding_model_name, embedding_dim
            )

            from langchain_community.vectorstores import Milvus 
            vector_db = Milvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
            )
            
            found_docs = vector_db.similarity_search(query, k=k)
            logging.info(f"Found {len(found_docs)} documents in collection '{collection_name}' for query.")
            return found_docs
        except Exception as e:
            logging.error(f"Error searching documents in Milvus collection '{collection_name}': {e}", exc_info=True)
            return []


    def get_all_sources_in_collection(self, collection_name: str) -> list[str]:
        """
        Lấy danh sách tất cả các tên nguồn (filenames) duy nhất trong một collection.
        """
        if not utility.has_collection(collection_name):
            return []
        
        collection = MilvusCollection(collection_name)
        collection.load()
        try:
            results = collection.query(expr="id >= 0", output_fields=["metadata"])
            unique_sources = sorted(list(set([res['metadata'].get('source') for res in results if 'source' in res.get('metadata', {})])))
            collection.release()
            return unique_sources
        except Exception as e:
            logging.error(f"Error getting all sources for collection '{collection_name}': {e}", exc_info=True)
            collection.release()
            return []

    def get_chunks_by_source(self, collection_name: str, source_filename: str) -> list[dict]:
        """
        Lấy tất cả các chunk (nội dung và metadata) cho một tên file cụ thể trong một collection.
        """
        if not utility.has_collection(collection_name):
            return []
        
        collection = MilvusCollection(collection_name)
        collection.load()
        expr = f'metadata["source"] == "{source_filename}"'
        try:
            results = collection.query(expr=expr, output_fields=["text", "metadata"])
            
            formatted_results = []
            for res in results:
                chunk_data = {
                    "id": str(res.get('id')),
                    "text": res.get('text'),
                    "source": res.get('metadata', {}).get('source'),
                    "metadata": res.get('metadata', {})
                }
                formatted_results.append(chunk_data)

            collection.release()
            return formatted_results
        except Exception as e:
            logging.error(f"Error getting chunks by source '{source_filename}' in collection '{collection_name}': {e}", exc_info=True)
            collection.release()
            return []

    def get_all_chunks(self, collection_name: str) -> list[dict]:
        if not utility.has_collection(collection_name):
            print("Entity count2:", collection.num_entities)
            return []
        collection = MilvusCollection(collection_name)
        collection.load()
        print("Entity count:", collection.num_entities)
        try:
            results = collection.query(expr="id >= 0", output_fields=["*"])
            chunks = []
            for r in results:
                print(r)  # 🕵️ Quan trọng để xem nó ra gì
                chunk = {
                    "id": r.get("id"),
                    "vector": r.get("embedding"),
                    "text": r.get("content") or r.get("text") or "N/A",
                    "metadata": r.get("metadata") or {}
                }
                chunks.append(chunk)
            return chunks
        finally:
            collection.release()



milvus_service = MilvusService()