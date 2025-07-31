# milvus_langchain.py
from typing import List, Dict, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MilvusService:
    """
    Service to manage interactions with Milvus, using Langchain for vector store operations.
    """
    def __init__(self, uri: str, embedding_function: Embeddings):
        """
        Initializes MilvusService.

        Args:
            uri (str): URI to connect to Milvus (e.g., "http://localhost:19530").
            embedding_function (Embeddings): The embedding function used to create vectors.
        """
        self.uri = uri
        self.embedding_function = embedding_function
        self._connect()
        # Cache for loaded Milvus Collection objects to avoid repeated load/release operations
        # for direct PyMilvus queries (e.g., show_all_data, get_all_documents).
        # Langchain Milvus (used by insert_documents, search_documents) manages its own internal state.
        self._loaded_collections: Dict[str, Collection] = {}
        logging.info(f"MilvusService initialized with URI: {uri}")

    def _connect(self):
        """
        Establishes connection to Milvus.
        """
        try:
            connections.connect(uri=self.uri)
            logging.info(f"Successfully connected to Milvus at {self.uri}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus at {self.uri}: {e}")
            raise

    def _get_collection_instance(self, collection_name: str) -> Optional[Collection]:
        """
        Retrieves a Milvus Collection instance from cache, loading it if not already loaded.
        This is used for direct PyMilvus operations that require a loaded collection.
        """
        if collection_name not in self._loaded_collections:
            if not utility.has_collection(collection_name):
                logging.warning(f"Collection '{collection_name}' does not exist.")
                return None
            try:
                collection = Collection(name=collection_name)
                collection.load() # Load the collection into memory
                self._loaded_collections[collection_name] = collection
                logging.info(f"Collection '{collection_name}' loaded into cache.")
            except Exception as e:
                logging.error(f"Error loading collection '{collection_name}': {e}")
                return None
        return self._loaded_collections[collection_name]

    def _release_collection_instance(self, collection_name: str):
        """
        Releases a Milvus Collection instance from memory and removes it from cache.
        Call this when a collection is no longer actively needed to free up resources,
        e.g., before dropping the collection.
        """
        if collection_name in self._loaded_collections:
            try:
                self._loaded_collections[collection_name].release() # Release from Milvus memory
                del self._loaded_collections[collection_name] # Remove from Python cache
                logging.info(f"Collection '{collection_name}' released from memory and cache.")
            except Exception as e:
                logging.error(f"Error releasing collection '{collection_name}': {e}")


    def _get_vectorstore(self, collection_name: str, drop_old: bool = False) -> Milvus:
        """
        Gets an instance of Langchain Milvus vector store.
        Note: Langchain Milvus internally handles connection and collection state for its operations
        (add_documents, similarity_search). This method is primarily for providing the Langchain wrapper.

        Args:
            collection_name (str): Name of the Milvus collection.
            drop_old (bool): If True, will drop the old collection if it exists. Defaults to False.

        Returns:
            Milvus: Instance of Langchain Milvus.
        """
        return Milvus(
            embedding_function=self.embedding_function,
            connection_args={"uri": self.uri},
            collection_name=collection_name,
            drop_old=drop_old
        )

    def create_collection(self, collection_name: str, vector_dim: int, recreate: bool = False) -> bool:
        """
        Creates a new collection in Milvus.

        Args:
            collection_name (str): Name of the collection to create.
            vector_dim (int): Dimension of the embedding vector.
            recreate (bool): If True, will drop the collection if it already exists and recreate it. Defaults to False.

        Returns:
            bool: True if the collection was created successfully or already exists, False if an error occurred.
        """
        try:
            # Validate collection name before proceeding
            # Milvus collection names can only contain letters, numbers, and underscores.
            if not collection_name.replace('_', '').isalnum():
                logging.error(f"Invalid collection name: '{collection_name}'. Collection names can only contain letters, numbers, and underscores.")
                return False

            if utility.has_collection(collection_name):
                if recreate:
                    logging.warning(f"Collection '{collection_name}' already exists. Dropping and recreating.")
                    self._release_collection_instance(collection_name) # Release before dropping
                    utility.drop_collection(collection_name)
                else:
                    logging.info(f"Collection '{collection_name}' already exists. Skipping creation.")
                    return True

            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # To store page_content
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256), # To store metadata source
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=128), # To store metadata content_type
                FieldSchema(name="page", dtype=DataType.INT64), # To store metadata page
                FieldSchema(name="total_pages", dtype=DataType.INT64) # To store metadata total_pages
            ]
            schema = CollectionSchema(fields, f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)

            # Create index for vector field
            # IVF_FLAT is a good starting point for ANN search.
            index_params = {
                "metric_type": "L2", # Euclidean distance
                "index_type": "IVF_FLAT", # Inverted File Index with Flat quantizer
                "params": {"nlist": 128} # Number of clusters
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"Collection '{collection_name}' created successfully with vector dimension {vector_dim}.")
            
            # Load the newly created collection into cache for immediate use
            collection.load() 
            self._loaded_collections[collection_name] = collection
            return True
        except Exception as e:
            logging.error(f"Error creating collection '{collection_name}': {e}")
            return False

    def insert_documents(self, collection_name: str, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Inserts documents into a Milvus collection.

        Args:
            collection_name (str): Name of the target collection.
            documents (List[Document]): List of Langchain Document objects to insert.
            ids (Optional[List[str]]): Optional list of IDs for each document. If not provided, Milvus
                                       will automatically generate IDs.

        Returns:
            List[str]: List of IDs of the inserted documents.
        """
        # No need to get collection instance and load/release here, Langchain Milvus handles it.
        # However, ensure the collection exists before attempting to insert.
        if not utility.has_collection(collection_name):
            logging.error(f"Collection '{collection_name}' does not exist. Cannot insert documents.")
            return []

        vectorstore = self._get_vectorstore(collection_name)
        try:
            inserted_ids = vectorstore.add_documents(documents=documents, ids=ids)
            logging.info(f"Successfully inserted {len(documents)} documents into collection '{collection_name}'.")
            return inserted_ids
        except Exception as e:
            logging.error(f"Error inserting documents into collection '{collection_name}': {e}")
            return []

    def search_documents(self, collection_name: str, query: str, k: int = 4) -> List[Document]:
        """
        Searches for similar documents in a Milvus collection.

        Args:
            collection_name (str): Name of the collection to search.
            query (str): The query string.
            k (int): Number of top similar documents to return.

        Returns:
            List[Document]: List of matching Langchain Document objects.
        """
        # No need to get collection instance and load/release here, Langchain Milvus handles it.
        # However, ensure the collection exists before attempting to search.
        if not utility.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' does not exist. Returning empty list for search.")
            return []

        vectorstore = self._get_vectorstore(collection_name)
        try:
            results = vectorstore.similarity_search(query, k=k)
            logging.info(f"Found {len(results)} documents for query in collection '{collection_name}'.")
            return results
        except Exception as e:
            logging.error(f"Error searching documents in collection '{collection_name}': {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """
        Deletes an entire Milvus collection.

        Args:
            collection_name (str): Name of the collection to delete.

        Returns:
            bool: True if the collection was deleted successfully, False if an error occurred.
        """
        try:
            if utility.has_collection(collection_name):
                self._release_collection_instance(collection_name) # Release from cache and Milvus memory before dropping
                utility.drop_collection(collection_name)
                logging.info(f"Collection '{collection_name}' deleted successfully.")
                return True
            else:
                logging.info(f"Collection '{collection_name}' does not exist. Nothing to delete.")
                return False
        except Exception as e:
            logging.error(f"Error deleting collection '{collection_name}': {e}")
            return False

    def show_all_data(self, collection_name: str, limit: int = 100) -> Dict[str, int]:
        """
        Displays all data (or a limited number) and counts documents by source.

        Args:
            collection_name (str): Name of the collection.
            limit (int): Maximum number of documents to query.

        Returns:
            Dict[str, int]: A dictionary containing the count of documents for each source.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return {} # Return empty if collection doesn't exist or couldn't be loaded

        try:
            # Query all records, limit the number to avoid loading too much data
            results = collection.query(expr="pk != ''", output_fields=["source"], limit=limit)
            
            source_counts = {}
            for res in results:
                source = res.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logging.info(f"Retrieved {len(results)} source counts for collection '{collection_name}'.") # Corrected logging
            return source_counts
        except Exception as e:
            logging.error(f"Error showing all data for collection '{collection_name}': {e}")
            return {}

    def get_document_by_id(self, collection_name: str, doc_id: str) -> Optional[Document]:
        """
        Retrieves a document by its ID.

        Args:
            collection_name (str): Name of the collection.
            doc_id (str): ID of the document to retrieve.

        Returns:
            Optional[Document]: Langchain Document object if found, otherwise None.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return None # Return None if collection doesn't exist or couldn't be loaded

        try:
            # Query document by pk (primary key)
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
                        'id': doc_id # Add ID to metadata for easy retrieval
                    }
                )
            logging.info(f"Document with ID '{doc_id}' not found in collection '{collection_name}'.")
            return None
        except Exception as e:
            logging.error(f"Error getting document by ID '{doc_id}' from collection '{collection_name}': {e}")
            return None

    def delete_by_id(self, collection_name: str, doc_id: str) -> bool:
        """
        Deletes a document from the collection by its ID.

        Args:
            collection_name (str): Name of the collection.
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if the document was deleted successfully, False if an error occurred or not found.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            logging.warning(f"Collection '{collection_name}' not found or loaded. Cannot delete by ID.")
            return False

        try:
            # Delete document based on primary key
            result = collection.delete(expr=f"pk == '{doc_id}'")
            collection.flush() # Ensure changes are written to disk
            
            if result.delete_count > 0:
                logging.info(f"Successfully deleted document with ID '{doc_id}' from collection '{collection_name}'.")
                return True
            else:
                logging.warning(f"Document with ID '{doc_id}' not found or not deleted in collection '{collection_name}'.")
                return False
        except Exception as e:
            logging.error(f"Error deleting document with ID '{doc_id}' from collection '{collection_name}': {e}")
            return False

    def delete_by_source(self, collection_name: str, source: str) -> bool:
        """
        Deletes all documents with the same source from the collection.

        Args:
            collection_name (str): Name of the collection.
            source (str): Name of the source to delete.

        Returns:
            bool: True if documents were deleted successfully, False if an error occurred.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            logging.warning(f"Collection '{collection_name}' not found or loaded. Cannot delete by source.")
            return False

        try:
            result = collection.delete(expr=f"source == '{source}'")
            collection.flush()
            if result.delete_count > 0:
                logging.info(f"Successfully deleted {result.delete_count} documents with source '{source}' from collection '{collection_name}'.")
                return True
            else:
                logging.warning(f"No documents found with source '{source}' in collection '{collection_name}' for deletion.")
                return False
        except Exception as e:
            logging.error(f"Error deleting documents with source '{source}' from collection '{collection_name}': {e}")
            return False

    def get_all_documents(self, collection_name: str, limit: int = 100) -> List[Document]:
        """
        Retrieves all documents (or a limited number) from a collection.

        Args:
            collection_name (str): Name of the collection.
            limit (int): Maximum number of documents to return.

        Returns:
            List[Document]: List of Langchain Document objects.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return [] # Return empty if collection doesn't exist or couldn't be loaded

        try:
            # Query all records, limit the number
            results = collection.query(
                expr="pk != ''",
                output_fields=["pk", "text", "source", "content_type", "page", "total_pages"],
                limit=limit
            )
            
            documents = []
            for res in results:
                documents.append(Document(
                    page_content=res.get('text', ''),
                    metadata={
                        'source': res.get('source', ''),
                        'content_type': res.get('content_type', ''),
                        'page': res.get('page', 1),
                        'total_pages': res.get('total_pages', 1),
                        'id': res.get('pk', '')
                    }
                ))
            logging.info(f"Retrieved {len(documents)} documents from collection '{collection_name}'.") # Corrected logging
            return documents
        except Exception as e:
            logging.error(f"Error getting all documents from collection '{collection_name}': {e}")
            return []

    def list_collections(self) -> List[str]:
        """
        Lists all existing collection names in Milvus.

        Returns:
            List[str]: A list of collection names.
        """
        try:
            collections = utility.list_collections()
            logging.info(f"Retrieved list of collections: {collections}")
            return collections
        except Exception as e:
            logging.error(f"Error listing collections: {e}")
            return []

    def describe_collection(self, collection_name: str) -> Optional[Dict]:
        """
        Describes a Milvus collection, returning its schema and other details.

        Args:
            collection_name (str): Name of the collection to describe.

        Returns:
            Optional[Dict]: A dictionary containing collection details, or None if not found/error.
        """
        if not utility.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' does not exist. Cannot describe.")
            return None
        try:
            collection = Collection(collection_name)
            # You might need to adjust what information is returned based on your needs
            # This is a simplified example.
            description = {
                "name": collection.name,
                "description": collection.description,
                "schema": [field.to_dict() for field in collection.schema.fields],
                "num_entities": collection.num_entities, # This might trigger load if not loaded
                "primary_field": collection.primary_field.name if collection.primary_field else None,
                "auto_id": collection.auto_id,
                "shards_num": collection.shards_num,
                "consistency_level": collection.consistency_level,
                "indexes": [index.to_dict() for index in collection.indexes] if collection.indexes else []
            }
            logging.info(f"Described collection '{collection_name}'.")
            return description
        except Exception as e:
            logging.error(f"Error describing collection '{collection_name}': {e}")
            return None

    def get_collection_stats(self, collection_name: str) -> Optional[Dict]:
        """
        Gets statistics for a Milvus collection, such as number of entities.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            Optional[Dict]: A dictionary containing collection statistics, or None if not found/error.
        """
        collection = self._get_collection_instance(collection_name)
        if not collection:
            return None

        try:
            # num_entities is a property that can be directly accessed on a loaded collection
            stats = {
                "name": collection.name,
                "num_entities": collection.num_entities,
                # You can add more stats if needed, e.g., disk usage, index size if exposed by PyMilvus
            }
            logging.info(f"Retrieved stats for collection '{collection_name}'.")
            return stats
        except Exception as e:
            logging.error(f"Error getting stats for collection '{collection_name}': {e}")
            return None

