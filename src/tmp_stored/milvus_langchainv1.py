# milvus_langchain.py
from typing import List, Dict, Optional, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MilvusService:
    """
    Dịch vụ để quản lý tương tác với Milvus, sử dụng Langchain cho các hoạt động lưu trữ vector.
    """
    def __init__(self, uri: str, embedding_function: Embeddings):
        """
        Khởi tạo MilvusService.

        Args:
            uri (str): URI để kết nối tới Milvus (ví dụ: "http://localhost:19530").
            embedding_function (Embeddings): Hàm nhúng được sử dụng để tạo vector.
        """
        self.uri = uri
        self.embedding_function = embedding_function
        self._connect()
        # Cache cho các đối tượng Milvus Collection đã tải để tránh lặp lại các hoạt động tải/giải phóng
        # cho các truy vấn PyMilvus trực tiếp (ví dụ: show_all_data, get_all_documents).
        # Langchain Milvus (được sử dụng bởi insert_documents, search_documents) quản lý trạng thái nội bộ của nó.
        self._loaded_collections: Dict[str, Collection] = {}
        logging.info(f"MilvusService đã được khởi tạo với URI: {uri}")

    def _connect(self):
        """
        Thiết lập kết nối tới Milvus.
        """
        try:
            connections.connect(uri=self.uri)
            logging.info(f"Đã kết nối thành công tới Milvus tại {self.uri}")
        except Exception as e:
            logging.error(f"Không thể kết nối tới Milvus tại {self.uri}: {e}")
            raise # Ném lại lỗi để đảm bảo khởi động thất bại nếu không thể thiết lập kết nối

    def _get_collection_instance(self, collection_name: str) -> Optional[Collection]:
        """
        Lấy một thể hiện của Milvus Collection từ bộ nhớ cache, tải nó nếu chưa được tải.
        Điều này được sử dụng cho các hoạt động PyMilvus trực tiếp yêu cầu một collection đã được tải.
        """
        if collection_name not in self._loaded_collections:
            if not utility.has_collection(collection_name):
                logging.warning(f"Collection '{collection_name}' không tồn tại.")
                return None
            try:
                collection = Collection(name=collection_name)
                collection.load() # Tải collection vào bộ nhớ
                self._loaded_collections[collection_name] = collection
                logging.info(f"Collection '{collection_name}' đã được tải vào bộ nhớ cache.")
            except Exception as e:
                logging.error(f"Lỗi khi tải collection '{collection_name}': {e}", exc_info=True)
                return None
        return self._loaded_collections[collection_name]

    def _release_collection_instance(self, collection_name: str):
        """
        Giải phóng một thể hiện của Milvus Collection khỏi bộ nhớ và xóa nó khỏi bộ nhớ cache.
        Gọi hàm này khi một collection không còn được sử dụng tích cực để giải phóng tài nguyên,
        ví dụ: trước khi xóa collection.
        """
        if collection_name in self._loaded_collections:
            try:
                self._loaded_collections[collection_name].release() # Giải phóng khỏi bộ nhớ Milvus
                del self._loaded_collections[collection_name] # Xóa khỏi bộ nhớ cache Python
                logging.info(f"Collection '{collection_name}' đã được giải phóng khỏi bộ nhớ và bộ nhớ cache.")
            except Exception as e:
                logging.error(f"Lỗi khi giải phóng collection '{collection_name}': {e}", exc_info=True)


    def _get_vectorstore(self, collection_name: str, drop_old: bool = False) -> Milvus:
        """
        Lấy một thể hiện của kho vector Langchain Milvus.
        Lưu ý: Langchain Milvus tự xử lý kết nối và trạng thái collection cho các hoạt động của nó
        (add_documents, similarity_search). Phương thức này chủ yếu để cung cấp trình bao bọc Langchain.

        Args:
            collection_name (str): Tên của collection Milvus.
            drop_old (bool): Nếu True, sẽ xóa collection cũ nếu nó tồn tại. Mặc định là False.

        Returns:
            Milvus: Thể hiện của Langchain Milvus.
        """
        return Milvus(
            embedding_function=self.embedding_function,
            connection_args={"uri": self.uri},
            collection_name=collection_name,
            drop_old=drop_old
        )

    def create_collection(self, collection_name: str, vector_dim: int, recreate: bool = False) -> bool:
        """
        Tạo một collection mới trong Milvus.

        Args:
            collection_name (str): Tên của collection cần tạo.
            vector_dim (int): Kích thước của vector nhúng.
            recreate (bool): Nếu True, sẽ xóa collection nếu nó đã tồn tại và tạo lại. Mặc định là False.

        Returns:
            bool: True nếu collection được tạo thành công hoặc đã tồn tại, False nếu có lỗi xảy ra.
        """
        try:
            # Xác thực tên collection trước khi tiếp tục
            # Tên collection Milvus chỉ có thể chứa chữ cái, số và dấu gạch dưới.
            if not collection_name.replace('_', '').isalnum():
                logging.error(f"Tên collection không hợp lệ: '{collection_name}'. Tên collection chỉ có thể chứa chữ cái, số và dấu gạch dưới.")
                return False

            if utility.has_collection(collection_name):
                if recreate:
                    logging.warning(f"Collection '{collection_name}' đã tồn tại. Đang xóa và tạo lại.")
                    self._release_collection_instance(collection_name) # Giải phóng trước khi xóa
                    utility.drop_collection(collection_name)
                else:
                    logging.info(f"Collection '{collection_name}' đã tồn tại. Bỏ qua việc tạo.")
                    return True

            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # Để lưu trữ page_content
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256), # Để lưu trữ metadata source (tên file)
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=128), # Để lưu trữ metadata content_type
                FieldSchema(name="page", dtype=DataType.INT64), # Để lưu trữ metadata page
                FieldSchema(name="total_pages", dtype=DataType.INT64) # Để lưu trữ metadata total_pages
            ]
            schema = CollectionSchema(fields, f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)

            # Tạo chỉ mục cho trường vector
            # IVF_FLAT là một điểm khởi đầu tốt cho tìm kiếm ANN.
            index_params = {
                "metric_type": "L2", # Khoảng cách Euclidean
                "index_type": "IVF_FLAT", # Chỉ mục tệp đảo ngược với bộ lượng tử phẳng
                "params": {"nlist": 128} # Số lượng cụm
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"Collection '{collection_name}' created successfully with vector dimension {vector_dim}.")
            # Load the newly created collection into cache for immediate use
            collection.load()
            self._loaded_collections[collection_name] = collection
            return True
        except Exception as e:
            logging.error(f"Lỗi khi tạo collection '{collection_name}': {e}", exc_info=True)
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Xóa toàn bộ Milvus collection.

        Args:
            collection_name (str): Tên của collection cần xóa.

        Returns:
            bool: True nếu collection được xóa thành công hoặc không tồn tại, False nếu có lỗi.
        """
        try:
            if utility.has_collection(collection_name):
                self._release_collection_instance(collection_name) # Giải phóng khỏi bộ nhớ cache và bộ nhớ Milvus trước khi xóa
                utility.drop_collection(collection_name)
                logging.info(f"Collection '{collection_name}' đã được xóa thành công.")
                return True
            else:
                logging.info(f"Collection '{collection_name}' không tồn tại. Không cần hành động.")
                return True
        except Exception as e:
            logging.error(f"Lỗi khi xóa collection '{collection_name}': {e}", exc_info=True)
            return False

    def add_documents(self, collection_name: str, documents: List[Document]) -> List[str]:
        """
        Thêm tài liệu vào một Milvus collection được chỉ định. Mỗi tài liệu phải có một 'pk' trong metadata của nó.

        Args:
            collection_name (str): Tên của collection để thêm tài liệu vào.
            documents (List[Document]): Một danh sách các đối tượng Langchain Document để chèn.
                                        Mỗi Document's metadata PHẢI chứa một trường 'pk'.

        Returns:
            List[str]: Một danh sách các khóa chính của các tài liệu đã chèn.
        """
        if not documents:
            logging.warning("Không có tài liệu nào được cung cấp để chèn.")
            return []

        # Đảm bảo collection tồn tại và được tải
        if not utility.has_collection(collection_name):
            logging.error(f"Collection '{collection_name}' không tồn tại. Không thể chèn tài liệu.")
            return []

        try:
            vector_store = self._get_vectorstore(collection_name)
            # Langchain Milvus's add_documents sẽ sử dụng 'pk' từ metadata nếu auto_id=False
            pks = vector_store.add_documents(documents)
            logging.info(f"Đã chèn thành công {len(documents)} tài liệu vào collection '{collection_name}'. PK đã chèn: {pks}")
            return pks
        except Exception as e:
            logging.error(f"Lỗi khi chèn tài liệu vào collection '{collection_name}': {e}", exc_info=True)
            return []

    def search_documents(self, collection_name: str, query: str, k: int = 4) -> List[Document]:
        """
        Thực hiện tìm kiếm tương tự trong Milvus collection được chỉ định.

        Args:
            collection_name (str): Tên của collection để tìm kiếm.
            query (str): Chuỗi truy vấn.
            k (int): Số lượng tài liệu tương tự hàng đầu để trả về.

        Returns:
            List[Document]: Một danh sách các đối tượng Langchain Document có liên quan.
        """
        try:
            vector_store = self._get_vectorstore(collection_name)
            docs = vector_store.similarity_search(query, k=k)
            logging.info(f"Đã tìm thấy {len(docs)} tài liệu cho truy vấn trong collection '{collection_name}' với k = '{k}'." )
            return docs
        except Exception as e:
            logging.error(f"Lỗi khi tìm kiếm tài liệu trong collection '{collection_name}': {e}", exc_info=True)
            return []

    def get_all_documents(self, collection_name: str) -> List[Document]:
        """
        Truy xuất tất cả tài liệu từ một Milvus collection được chỉ định.
        Thao tác này có thể tốn nhiều bộ nhớ đối với các collection lớn.
        """
        try:
            collection = self._get_collection_instance(collection_name)
            if not collection:
                return []

            expr = "pk like '%%'" # Biểu thức đơn giản để truy xuất tất cả các thực thể
            # Giới hạn các trường đầu ra để tránh tìm nạp các nhúng nếu không cần thiết cho trường hợp sử dụng
            output_fields = ["pk", "text", "source", "content_type", "page", "total_pages"]
            results = collection.query(expr=expr, output_fields=output_fields, consistency_level="Strong")

            documents = []
            for item in results:
                # Xây dựng lại Langchain Document từ thực thể Milvus
                metadata = {
                    "pk": item.get("pk"),
                    "source": item.get("source"),
                    "content_type": item.get("content_type"),
                    "page": item.get("page"),
                    "total_pages": item.get("total_pages"),
                }
                # Lọc các giá trị None và chỉ bao gồm các trường metadata hiện có
                metadata = {k: v for k, v in metadata.items() if v is not None}
                documents.append(Document(page_content=item.get("text", ""), metadata=metadata))
            logging.info(f"Đã truy xuất {len(documents)} tài liệu từ collection '{collection_name}'.")
            return documents
        except Exception as e:
            logging.error(f"Lỗi khi truy xuất tất cả tài liệu từ collection '{collection_name}': {e}", exc_info=True)
            return []

    def get_document_count_by_source(self, collection_name: str, source: str) -> int:
        """
        Đếm số lượng tài liệu liên quan đến một nguồn cụ thể (tên tệp) trong một Milvus collection.

        Args:
            collection_name (str): Tên của collection.
            source (str): Giá trị trường metadata 'source' (ví dụ: tên tệp).

        Returns:
            int: Số lượng tài liệu được tìm thấy cho nguồn đã cho.
        """
        try:
            collection = self._get_collection_instance(collection_name)
            if not collection:
                logging.warning(f"Collection '{collection_name}' không tìm thấy hoặc không được tải.")
                return 0

            # Xây dựng biểu thức truy vấn để lọc theo nguồn
            # Đảm bảo thoát đúng cách cho nguồn nếu nó chứa các ký tự đặc biệt,
            # nhưng đối với tên tệp, thường thì khớp chuỗi đơn giản là ổn.
            # Chuỗi truy vấn Milvus cho VARCHAR cần dấu nháy đơn xung quanh các hằng số chuỗi.
            expr = f"source == '{source}'"
            
            results = collection.query(expr=expr, output_fields=["pk"], consistency_level="Strong")
            count = len(results)
            logging.info(f"Đã tìm thấy {count} tài liệu cho nguồn '{source}' trong collection '{collection_name}'.")
            return count
        except Exception as e:
            logging.error(f"Lỗi khi đếm tài liệu cho nguồn '{source}' trong collection '{collection_name}': {e}", exc_info=True)
            return 0

    def get_documents_by_source(self, collection_name: str, source: str, limit: int = 100) -> List[Document]:
        """
        Truy xuất tài liệu từ một Milvus collection dựa trên tên nguồn (source) và giới hạn số lượng.

        Args:
            collection_name (str): Tên của collection.
            source (str): Tên nguồn để lọc tài liệu.
            limit (int): Số lượng tài liệu tối đa cần trả về.

        Returns:
            List[Document]: Danh sách các đối tượng Langchain Document phù hợp.
        """
        try:
            collection = self._get_collection_instance(collection_name)
            if not collection:
                logging.warning(f"Collection '{collection_name}' không tìm thấy hoặc không được tải.")
                return []

            expr = f"source == '{source}'"
            output_fields = ["pk", "text", "source", "content_type", "page", "total_pages"]
            
            # Milvus query does not directly support LIMIT in expr, so we fetch all and slice
            # For very large datasets, this might be inefficient. Consider batching or
            # more advanced Milvus query features if performance becomes an issue.
            results = collection.query(
                expr=expr,
                output_fields=output_fields,
                consistency_level="Strong",
                limit=limit # PyMilvus query supports limit directly
            )

            documents = []
            for item in results:
                metadata = {
                    "pk": item.get("pk"),
                    "source": item.get("source"),
                    "content_type": item.get("content_type"),
                    "page": item.get("page"),
                    "total_pages": item.get("total_pages"),
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}
                documents.append(Document(page_content=item.get("text", ""), metadata=metadata))
            
            logging.info(f"Đã truy xuất {len(documents)} tài liệu từ nguồn '{source}' trong collection '{collection_name}'.")
            return documents
        except Exception as e:
            logging.error(f"Lỗi khi truy xuất tài liệu theo nguồn '{source}' trong collection '{collection_name}': {e}", exc_info=True)
            return []

    def search_by_metadata(self, collection_name: str, metadata_filters: Dict[str, Any], limit: int = 100) -> List[Document]:
        """
        Tìm kiếm tài liệu trong Milvus collection dựa trên các bộ lọc metadata.

        Args:
            collection_name (str): Tên của collection.
            metadata_filters (Dict[str, Any]): Từ điển chứa các cặp key-value của metadata để lọc.
                                                Ví dụ: {"content_type": "application/pdf", "page": 5}
            limit (int): Số lượng tài liệu tối đa cần trả về.

        Returns:
            List[Document]: Danh sách các đối tượng Langchain Document phù hợp.
        """
        try:
            collection = self._get_collection_instance(collection_name)
            if not collection:
                logging.warning(f"Collection '{collection_name}' không tìm thấy hoặc không được tải.")
                return []

            # Xây dựng biểu thức truy vấn từ metadata_filters
            # Lưu ý: Các trường metadata phải được định nghĩa trong schema của Milvus collection
            # và phải có kiểu dữ liệu phù hợp để truy vấn.
            filter_parts = []
            for key, value in metadata_filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} == '{value}'")
                elif isinstance(value, (int, float)):
                    filter_parts.append(f"{key} == {value}")
                # Thêm các loại dữ liệu khác nếu cần (ví dụ: boolean)
                else:
                    logging.warning(f"Loại dữ liệu không được hỗ trợ cho bộ lọc metadata: {key}: {type(value)}")
                    continue
            
            if not filter_parts:
                logging.warning("Không có bộ lọc metadata hợp lệ nào được cung cấp.")
                return []

            expr = " and ".join(filter_parts)
            output_fields = ["pk", "text", "source", "content_type", "page", "total_pages"]

            results = collection.query(
                expr=expr,
                output_fields=output_fields,
                consistency_level="Strong",
                limit=limit # PyMilvus query supports limit directly
            )

            documents = []
            for item in results:
                metadata = {
                    "pk": item.get("pk"),
                    "source": item.get("source"),
                    "content_type": item.get("content_type"),
                    "page": item.get("page"),
                    "total_pages": item.get("total_pages"),
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}
                documents.append(Document(page_content=item.get("text", ""), metadata=metadata))
            
            logging.info(f"Đã tìm thấy {len(documents)} tài liệu với bộ lọc metadata '{metadata_filters}' trong collection '{collection_name}'.")
            return documents
        except Exception as e:
            logging.error(f"Lỗi khi tìm kiếm tài liệu theo metadata trong collection '{collection_name}': {e}", exc_info=True)
            return []

    def get_document_by_id(self, collection_name: str, doc_id: str) -> Optional[Document]:
        """
        Truy xuất một tài liệu bằng ID của nó.

        Args:
            collection_name (str): Tên của collection.
            doc_id (str): ID của tài liệu cần truy xuất.

        Returns:
            Optional[Document]: Đối tượng Langchain Document nếu tìm thấy, ngược lại là None.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return None # Trả về None nếu collection không tồn tại hoặc không thể tải

        try:
            # Truy vấn tài liệu bằng pk (khóa chính)
            results = collection.query(expr=f"pk == '{doc_id}'", output_fields=["text", "source", "content_type", "page", "total_pages"])
            
            if results:
                doc_data = results[0]
                return Document(
                    page_content=doc_data.get('text', ''),
                    metadata={
                        'source': doc_data.get('source', ''),
                        'content_type': doc_data.get('content_type', ''),
                        'page': doc_data.get('page', 1),
                        'total_pages': doc_data.get('total_pages', 1),
                        'id': doc_id # Thêm ID vào metadata để dễ dàng truy xuất
                    }
                )
            logging.info(f"Tài liệu có ID '{doc_id}' không tìm thấy trong collection '{collection_name}'.")
            return None
        except Exception as e:
            logging.error(f"Lỗi khi lấy tài liệu bằng ID '{doc_id}' từ collection '{collection_name}': {e}", exc_info=True)
            return None

    def delete_by_id(self, collection_name: str, doc_id: str) -> bool:
        """
        Xóa một tài liệu khỏi collection bằng ID của nó.

        Args:
            collection_name (str): Tên của collection.
            doc_id (str): ID của tài liệu cần xóa.

        Returns:
            bool: True nếu tài liệu được xóa thành công, False nếu có lỗi xảy ra hoặc không tìm thấy.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            logging.warning(f"Collection '{collection_name}' không tìm thấy hoặc không được tải. Không thể xóa bằng ID.")
            return False

        try:
            # Xóa tài liệu dựa trên khóa chính
            result = collection.delete(expr=f"pk == '{doc_id}'")
            collection.flush() # Đảm bảo các thay đổi được ghi vào đĩa
            
            if result.delete_count > 0:
                logging.info(f"Đã xóa thành công tài liệu có ID '{doc_id}' khỏi collection '{collection_name}'.")
                return True
            else:
                logging.warning(f"Tài liệu có ID '{doc_id}' không tìm thấy hoặc không được xóa trong collection '{collection_name}'.")
                return False
        except Exception as e:
            logging.error(f"Lỗi khi xóa tài liệu có ID '{doc_id}' khỏi collection '{collection_name}': {e}", exc_info=True)
            return False

    def delete_by_source(self, collection_name: str, source: str) -> bool:
        """
        Xóa tất cả tài liệu có cùng nguồn khỏi collection.

        Args:
            collection_name (str): Tên của collection.
            source (str): Tên của nguồn cần xóa.

        Returns:
            bool: True nếu tài liệu được xóa thành công, False nếu có lỗi xảy ra.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            logging.warning(f"Collection '{collection_name}' không tìm thấy hoặc không được tải. Không thể xóa bằng nguồn.")
            return False

        try:
            result = collection.delete(expr=f"source == '{source}'")
            collection.flush()
            if result.delete_count > 0:
                logging.info(f"Đã xóa thành công {result.delete_count} tài liệu có nguồn '{source}' khỏi collection '{collection_name}'.")
                return True
            else:
                logging.warning(f"Không tìm thấy tài liệu nào có nguồn '{source}' trong collection '{collection_name}' để xóa.")
                return False
        except Exception as e:
            logging.error(f"Lỗi khi xóa tài liệu có nguồn '{source}' khỏi collection '{collection_name}': {e}", exc_info=True)
            return False

    def list_collections(self) -> List[str]:
        """
        Liệt kê tất cả các tên collection hiện có trong Milvus.

        Returns:
            List[str]: Một danh sách các tên collection.
        """
        try:
            collections = utility.list_collections()
            logging.info(f"Đã truy xuất danh sách các collection: {collections}")
            return collections
        except Exception as e:
            logging.error(f"Lỗi khi liệt kê các collection: {e}", exc_info=True)
            return []

    def describe_collection(self, collection_name: str) -> Optional[Dict]:
        """
        Mô tả một Milvus collection, trả về schema và các chi tiết khác.

        Args:
            collection_name (str): Tên của collection cần mô tả.

        Returns:
            Optional[Dict]: Một từ điển chứa chi tiết collection, hoặc None nếu không tìm thấy/lỗi.
        """
        if not utility.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' không tồn tại. Không thể mô tả.")
            return None
        try:
            collection = Collection(collection_name)
            # Load the collection to ensure schema is accessible
            collection.load()
            
            # Access schema properties directly
            description = {
                "name": collection.name,
                "description": collection.description,
                "schema": [field.to_dict() for field in collection.schema.fields],
                "num_entities": collection.num_entities,
                "primary_field": collection.schema.primary_field.name if collection.schema.primary_field else None,
                "auto_id": collection.schema.auto_id, # Corrected access
                "shards_num": collection.shards_num,
                "consistency_level": collection.consistency_level,
                "indexes": [index.to_dict() for index in collection.indexes] if collection.indexes else []
            }
            # Release the collection after describing to free up memory
            collection.release()
            logging.info(f"Đã mô tả collection '{collection_name}'.")
            return description
        except Exception as e:
            logging.error(f"Lỗi khi mô tả collection '{collection_name}': {e}", exc_info=True)
            return None

    def get_collection_stats(self, collection_name: str) -> Optional[Dict]:
        """
        Lấy thống kê cho một Milvus collection, chẳng hạn như số lượng thực thể.

        Args:
            collection_name (str): Tên của collection.

        Returns:
            Optional[Dict]: Một từ điển chứa thống kê collection, hoặc None nếu không tìm thấy/lỗi.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return None

        try:
            # num_entities là một thuộc tính có thể được truy cập trực tiếp trên một collection đã tải
            stats = {
                "name": collection.name,
                "num_entities": collection.num_entities,
                # Bạn có thể thêm nhiều thống kê hơn nếu cần, ví dụ: mức sử dụng đĩa, kích thước chỉ mục nếu được PyMilvus hiển thị
            }
            logging.info(f"Đã truy xuất thống kê cho collection '{collection_name}'.")
            return stats
        except Exception as e:
            logging.error(f"Lỗi khi lấy thống kê cho collection '{collection_name}': {e}", exc_info=True)
            return None
