# milvus_api_service.py
import os
import json
from flask import Flask, request, jsonify, send_from_directory,  render_template
from dotenv import load_dotenv
import logging
from flasgger import Swagger, swag_from
from flask_cors import CORS # Thêm dòng này

# Load environment variables from .env file
load_dotenv('./.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import components from created files
from src.milvus_langchain import MilvusService
from src.load_file import get_embedding_model # load_documents not used directly

app = Flask(
    __name__,
    # static_folder="../UI/Thien-ChatGUIV2",
    # template_folder="UI/Thien-ChatGUIV2"
    # sửa vì khi chạy ta chạy trong scr phải đi ra ngoài
    template_folder="../UI/Thien-ChatGUIV2"

)

swagger = Swagger(app)

CORS(app)  # Enable CORS for all routes

# Initialize MilvusService and embedding model - Singleton only create once when starting the service
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embeddings = get_embedding_model(
    provider=os.getenv("SERVER_EMBEDDING_PROVIDER", "openai"),
    model=EMBEDDING_MODEL
)
milvus_service = MilvusService(
    uri=os.getenv('MILVUS_URL', 'http://localhost:19530'),
    embedding_function=embeddings
)
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'test_data2')
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 1536)) 

try:
    milvus_service.create_collection(MILVUS_COLLECTION_NAME, VECTOR_DIMENSION, recreate=False)
    logging.info(f"Milvus collection '{MILVUS_COLLECTION_NAME}' ensured to exist.")
except Exception as e:
    logging.error(f"Failed to ensure Milvus collection '{MILVUS_COLLECTION_NAME}' exists: {e}")
    # It's critical to exit or handle gracefully if the core service dependency fails at startup
    exit(1)

@app.route('/milvus/collections/create', methods=['POST'])
@swag_from({
    'tags': ['Milvus Collections'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'collection_name': {'type': 'string'},
                    'vector_dim': {'type': 'integer'},
                    'recreate': {'type': 'boolean'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Collection created or ensured'}
    }
})
def create_milvus_collection():
    data = request.json
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)
    vector_dim = data.get('vector_dim', VECTOR_DIMENSION)
    recreate = data.get('recreate', False)

    if not collection_name or not vector_dim:
        return jsonify({'error': 'Collection name and vector dimension are required.'}), 400

    success = milvus_service.create_collection(collection_name, vector_dim, recreate)
    if success:
        return jsonify({'status': 'success', 'message': f"Collection '{collection_name}' created/ensured."})
    else:
        return jsonify({'status': 'error', 'message': f"Failed to create collection '{collection_name}'."}), 500

@app.route('/milvus/collections/delete', methods=['POST'])
@swag_from({
    'tags': ['Milvus Collections'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'collection_name': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Collection deleted'}
    }
})
def delete_milvus_collection():
    data = request.json
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)

    if not collection_name:
        return jsonify({'error': 'Collection name is required.'}), 400

    success = milvus_service.delete_collection(collection_name)
    if success:
        return jsonify({'status': 'success', 'message': f"Collection '{collection_name}' deleted."})
    else:
        return jsonify({'status': 'error', 'message': f"Failed to delete collection '{collection_name}' or it does not exist."}), 500

# Endpoint to list all Milvus collections
@app.route('/milvus/collections/list', methods=['GET'])
@swag_from({
    'tags': ['Milvus Collections'],
    'responses': {
        200: {'description': 'List of collections returned'}
    }
})
def list_milvus_collections():
    try:
        collections = milvus_service.list_collections()
        return jsonify({'status': 'success', 'collections': collections})
    except Exception as e:
        logging.error(f"Error listing Milvus collections: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Failed to list collections: {e}"}), 500

# Endpoint to describe a Milvus collection
@app.route('/milvus/collections/describe', methods=['GET'])
@swag_from({
    'tags': ['Milvus Collections'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Name of the collection to describe'}
    ],
    'responses': {
        200: {'description': 'Collection description returned'}
    }
})
def describe_milvus_collection():
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Collection name is required.'}), 400
    try:
        description = milvus_service.describe_collection(collection_name)
        if description:
            return jsonify({'status': 'success', 'description': description})
        else:
            return jsonify({'status': 'error', 'message': f"Collection '{collection_name}' not found or could not be described."}), 404
    except Exception as e:
        logging.error(f"Error describing Milvus collection '{collection_name}': {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Failed to describe collection '{collection_name}': {e}"}), 500

# Endpoint to get stats for a Milvus collection
@app.route('/milvus/collections/stats', methods=['GET'])
@swag_from({
    'tags': ['Milvus Collections'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Name of the collection to get stats for'}
    ],
    'responses': {
        200: {'description': 'Collection statistics returned'}
    }
})
def get_milvus_collection_stats():
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Collection name is required.'}), 400
    try:
        stats = milvus_service.get_collection_stats(collection_name)
        if stats:
            return jsonify({'status': 'success', 'stats': stats})
        else:
            return jsonify({'status': 'error', 'message': f"Collection '{collection_name}' not found or could not get stats."}), 404
    except Exception as e:
        logging.error(f"Error getting stats for Milvus collection '{collection_name}': {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Failed to get stats for collection '{collection_name}': {e}"}), 500


@app.route('/milvus/documents/insert', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'documents': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'page_content': {'type': 'string'},
                                'metadata': {'type': 'object'}
                            }
                        }
                    },
                    'collection_name': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Documents inserted'}
    }
})
def insert_milvus_documents():
    data = request.json
    documents_data = data.get('documents')
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)

    if not documents_data or not isinstance(documents_data, list):
        return jsonify({'error': 'A list of documents is required.'}), 400

    from langchain_core.documents import Document
    langchain_documents = []
    for doc_data in documents_data:
        if 'page_content' in doc_data:
            langchain_documents.append(Document(
                page_content=doc_data['page_content'],
                metadata=doc_data.get('metadata', {})
            ))
        else:
            return jsonify({'error': 'Each document must have "page_content".'}), 400

    try:
        inserted_ids = milvus_service.add_documents(collection_name, langchain_documents)
        if inserted_ids:
            return jsonify({'status': 'success', 'message': f"Inserted {len(inserted_ids)} documents.", 'ids': inserted_ids})
        else:
            # If add_documents returns an empty list but no exception, it means
            # an internal error occurred that was caught and logged within MilvusService.
            return jsonify({'status': 'error', 'message': "Failed to insert documents. Check Milvus API service logs for details."}), 500
    except Exception as e:
        logging.error(f"Error in insert_milvus_documents endpoint: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Internal server error during document insertion: {e}"}), 500

@app.route('/milvus/documents/search', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'k': {'type': 'integer'},
                    'collection_name': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Search results returned'}
    }
})
def search_milvus_documents():
    data = request.json
    query = data.get('query')
    k = data.get('k', int(os.getenv('SEARCH_K_VALUE', 4)))
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)

    if not query:
        return jsonify({'error': 'Query is required.'}), 400

    results = milvus_service.search_documents(collection_name, query, k)
    json_results = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in results]
    return jsonify({'status': 'success', 'results': json_results})

@app.route('/milvus/documents/get_by_id/<doc_id>', methods=['GET'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'doc_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Document ID to retrieve'
        },
        {
            'name': 'collection_name',
            'in': 'query',
            'type': 'string',
            'required': False
        }
    ],
    'responses': {
        200: {'description': 'Document retrieved'}
    }
})
def get_milvus_document_by_id(doc_id): # doc_id is now correctly passed as a function argument
    collection_name = request.args.get('collection_name', MILVUS_COLLECTION_NAME)
    doc = milvus_service.get_document_by_id(collection_name, doc_id)
    if doc:
        return jsonify({'status': 'success', 'document': {'page_content': doc.page_content, 'metadata': doc.metadata}})
    else:
        return jsonify({'status': 'error', 'message': f"Document with ID '{doc_id}' not found."}), 404

@app.route('/milvus/documents/delete_by_id', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'collection_name': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Document deleted'}
    }
})
def delete_milvus_document_by_id():
    data = request.json
    doc_id = data.get('id')
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)

    if not doc_id:
        return jsonify({'error': 'Document ID is required.'}), 400

    success = milvus_service.delete_by_id(collection_name, doc_id)
    if success:
        return jsonify({'status': 'success', 'message': f"Document with ID '{doc_id}' deleted."})
    else:
        return jsonify({'status': 'error', 'message': f"Failed to delete document with ID '{doc_id}' or it was not found."}), 500

@app.route('/milvus/documents/delete_by_source', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'source': {'type': 'string'},
                    'collection_name': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Documents deleted by source'}
    }
})
def delete_milvus_documents_by_source():
    data = request.json
    source = data.get('source')
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)

    if not source:
        return jsonify({'error': 'Source name is required.'}), 400

    success = milvus_service.delete_by_source(collection_name, source)
    if success:
        return jsonify({'status': 'success', 'message': f"Documents with source '{source}' deleted."})
    else:
        return jsonify({'status': 'error', 'message': f"Failed to delete documents with source '{source}' or none were found."}), 500

@app.route('/milvus/data/show_all', methods=['GET'])
@swag_from({
    'tags': ['Milvus Data'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string'},
        {'name': 'limit', 'in': 'query', 'type': 'integer'}
    ],
    'responses': {
        200: {'description': 'Source counts returned'}
    }
})
def show_all_milvus_data():
    collection_name = request.args.get('collection_name', MILVUS_COLLECTION_NAME)
    # Correctly use get_all_documents and then count sources
    all_docs = milvus_service.get_all_documents(collection_name)
    source_counts = {}
    for doc in all_docs:
        source = doc.metadata.get('source')
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    # The 'limit' parameter for show_all_data typically means limiting the number of sources returned,
    # not the number of documents fetched. For now, we return all source counts.
    # If a strict limit on sources is needed, additional logic would be required here.
    
    return jsonify({'status': 'success', 'source_counts': source_counts})

@app.route('/milvus/data/get_all_documents', methods=['GET'])
@swag_from({
    'tags': ['Milvus Data'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string'},
        {'name': 'limit', 'in': 'query', 'type': 'integer'}
    ],
    'responses': {
        200: {'description': 'All documents returned'}
    }
})
def get_all_milvus_documents():
    collection_name = request.args.get('collection_name', MILVUS_COLLECTION_NAME)
    limit = int(request.args.get('limit', 100))
    # MilvusService.get_all_documents fetches all documents.
    # We apply the limit here for the API response.
    docs = milvus_service.get_all_documents(collection_name)
    
    json_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in docs[:limit]]
    return jsonify({'status': 'success', 'documents': json_docs})

# NEW: Endpoint to get documents by source with limit
@app.route('/milvus/data/get_by_source', methods=['GET'])
@swag_from({
    'tags': ['Milvus Data'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Name of the collection'},
        {'name': 'source', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Source (filename) to filter by'},
        {'name': 'limit', 'in': 'query', 'type': 'integer', 'required': False, 'default': 100, 'description': 'Maximum number of documents to return'}
    ],
    'responses': {
        200: {'description': 'Documents filtered by source returned'}
    }
})
def get_milvus_data_by_source():
    collection_name = request.args.get('collection_name', MILVUS_COLLECTION_NAME)
    source = request.args.get('source')
    limit = int(request.args.get('limit', 100))

    if not source:
        return jsonify({'error': 'Source name is required.'}), 400

    docs = milvus_service.get_documents_by_source(collection_name, source, limit)
    json_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in docs]
    return jsonify({'status': 'success', 'documents': json_docs})

# NEW: Endpoint to search documents by metadata filters
@app.route('/milvus/documents/search_by_metadata', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'collection_name': {'type': 'string'},
                    'metadata_filters': {'type': 'object', 'description': 'Dictionary of metadata key-value pairs to filter by'},
                    'limit': {'type': 'integer', 'default': 100, 'description': 'Maximum number of documents to return'}
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Documents filtered by metadata returned'}
    }
})
def search_milvus_documents_by_metadata():
    data = request.json
    collection_name = data.get('collection_name', MILVUS_COLLECTION_NAME)
    metadata_filters = data.get('metadata_filters')
    limit = int(data.get('limit', 100))

    if not metadata_filters or not isinstance(metadata_filters, dict):
        return jsonify({'error': 'A dictionary of metadata_filters is required.'}), 400

    docs = milvus_service.search_by_metadata(collection_name, metadata_filters, limit)
    json_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in docs]
    return jsonify({'status': 'success', 'documents': json_docs})

@app.route('/')
def serve_ui():
    return render_template('index.html')
    
# thêm path file name để truy cập cùng câp với UI 
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.template_folder, filename)


if __name__ == '__main__':
    milvus_api_port = int(os.getenv("MILVUS_API_PORT", 5000))
    app.run(threaded=True, port=milvus_api_port, host='127.0.0.1')
