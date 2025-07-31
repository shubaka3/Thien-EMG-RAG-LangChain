# main.py
import time
import os
import json
import uuid
from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv, set_key, find_dotenv
import requests # Add requests library to call API
import logging
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
from flask_cors import CORS # Thêm dòng này

# Load environment variables for the first time
load_dotenv('./.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.agent import invoke_agent, stream_agent_response # agent.py still uses MilvusService directly
from src.load_file import process_single_document, move_file, get_embedding_model # Import new helper functions and get_embedding_model
from src.milvus_langchain import MilvusService # For direct interaction with Milvus

app = Flask(__name__)
swagger = Swagger(app) 
# Initialize Swagger
CORS(app)
# Configure file storage paths (these are read once at startup, changes require restart for paths to be re-evaluated)
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", './uploads')
PENDING_FILES_PATH = os.getenv("PENDING_FILES_PATH", './pending_files')
PROCESSED_FILES_PATH = os.getenv("PROCESSED_FILES_PATH", './processed_storage')
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'test_data2') # Default Milvus Collection name

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PENDING_FILES_PATH, exist_ok=True)
os.makedirs(PROCESSED_FILES_PATH, exist_ok=True)

# Initialize MilvusService and embedding model
# These variables are read once at startup for initialization.
# Changes to them require re-initialization of these objects, hence a server restart.
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embeddings = get_embedding_model(
    provider=os.getenv("SERVER_EMBEDDING_PROVIDER", "openai"),
    model=EMBEDDING_MODEL
)
milvus_service = MilvusService(
    uri=os.getenv('MILVUS_URL', 'http://localhost:19530'),
    embedding_function=embeddings
)

# --- Helper function to make authenticated requests to Milvus API Service ---
def _make_milvus_api_request(method: str, endpoint: str, json_data: dict = None, params: dict = None):
    # Read MILVUS_API_BASE_URL and MILVUS_API_KEY dynamically here
    MILVUS_API_BASE_URL_DYNAMIC = os.getenv("MILVUS_API_BASE_URL", "http://127.0.0.1:5000/milvus")
    MILVUS_API_KEY_DYNAMIC = os.getenv("MILVUS_API_KEY")

    headers = {}
    if MILVUS_API_KEY_DYNAMIC: 
        headers['X-API-Key'] = MILVUS_API_KEY_DYNAMIC
    
    url = f"{MILVUS_API_BASE_URL_DYNAMIC}{endpoint}"
    
    try:
        if method.lower() == 'get':
            response = requests.get(url, headers=headers, params=params)
        elif method.lower() == 'post':
            response = requests.post(url, headers=headers, json=json_data, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API at {url}: {e}")
        # Re-raise or handle appropriately
        raise

def make_response_chunk(chunk: str) -> str:
    """
    Creates a response chunk in the format expected by OpenAI API.

    Args:
        chunk (str): The content of the chunk.

    Returns:
        str: The serialized JSON string.
    """
    data = {
        'id': f"emgcmpl-{uuid.uuid4()}",
        'created': int(time.time()),
        'model': os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'), # This is read dynamically in agent.py now
        'choices': [{
            'index': 0,
            'logprobs': None,
            'finish_reason': None,
            'delta': {}
        }],
        'object': 'chat.completion.chunk',
    }
    if chunk is not None:
        data['choices'][0]['delta']['content'] = chunk
    return json.dumps(data)

@app.route('/api/chat/completion', methods=['POST'])
def chat():
    """
    Handles chat requests, which can be streaming or non-streaming.
    """
    data:dict = request.json
    
    if 'messages' not in data or not isinstance(data['messages'], list):
        return jsonify({'error': 'Message is required'}), 400
    question = data['messages'][0].get('content', '')
    if not question:
        return jsonify({'error': 'User question is required'}), 400

    if data.get('stream', False):
        @stream_with_context
        def generate():
            for chunk in stream_agent_response(question):
                yield f"data: {make_response_chunk(chunk)}\n\n"
            yield f"data: {make_response_chunk(None)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(generate(), mimetype='text/event-stream')
    else:
        answer = invoke_agent(question)
        return jsonify({
            'sources': [], # Can add sources here if the agent returns them
            'id': f'emgcmpl-{uuid.uuid4()}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'), # This is read dynamically in agent.py now
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': answer['answer']
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(question.split()),
                'completion_tokens': len(answer['answer'].split()),
                'total_tokens': len(answer['answer'].split())
            },
        })

@app.route('/api/sources', methods=['GET'])
def sources():
    """
    Retrieves a list of document sources and their counts from the Milvus API Service.
    This endpoint still uses the Milvus API Service.
    """
    try:
        response_data = _make_milvus_api_request('get', '/data/show_all')
        sources_data = response_data.get('source_counts', {})
        return jsonify([{'source': source, 'split_count': count} for source, count in sources_data.items()])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API for sources: {e}")
        return jsonify({'error': f'Failed to fetch sources from Milvus API: {e}'}), 500

@app.route('/api/upload_file', methods=['POST'])
@swag_from({
    'tags': ['File Management'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'File to upload (PDF, Markdown, JSON).'
        }
    ],
    'responses': {
        200: {'description': 'File uploaded to pending directory for processing'},
        400: {'description': 'No file part or disallowed file'},
        500: {'description': 'Error saving file'}
    }
})
def upload_file():
    """
    Uploads a file to the pending directory for later processing into Milvus.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        pending_file_path = os.path.join(UPLOAD_FOLDER, filename) # Use UPLOAD_FOLDER for initial upload
        processed_file_path = os.path.join(PROCESSED_FILES_PATH, filename)

        # Check if a file with the same name already exists in pending or processed storage
        if os.path.exists(pending_file_path):
            return jsonify({'message': f'File {filename} already exists in pending directory. Skipping upload.'}), 200
        if os.path.exists(processed_file_path):
            return jsonify({'message': f'File {filename} already exists in processed storage. Skipping upload.'}), 200

        try:
            file.save(pending_file_path)
            logging.info(f"File '{filename}' saved to '{UPLOAD_FOLDER}'.")
            return jsonify({'message': f'File {filename} uploaded to pending directory for processing.'}), 200
        except Exception as e:
            logging.error(f"Error saving uploaded file '{filename}': {e}")
            return jsonify({'error': f'Could not save file: {e}'}), 500

@app.route('/api/sources/update', methods=['POST'])
@swag_from({
    'tags': ['Milvus Documents'],
    'summary': 'Processes and indexes new documents from the pending directory into Milvus.',
    'description': 'Scans the pending directory for new documents, checks if they already exist in Milvus, '
                   'indexes new documents, and moves processed files to the storage directory. '
                   'Only files not already in Milvus are inserted.',
    'responses': {
        200: {'description': 'Document processing started successfully'},
        500: {'description': 'Error during document processing'}
    }
})
def update_source():
    """
    Processes documents from the pending directory, indexes them into Milvus,
    and moves them to the processed storage directory.
    """
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    files_to_process = os.listdir(PENDING_FILES_PATH)
    
    if not files_to_process:
        return jsonify({'message': 'No new files in pending directory to process.'}), 200

    logging.info(f"Starting to process {len(files_to_process)} files from '{PENDING_FILES_PATH}'...")

    for filename in files_to_process:
        file_path = os.path.join(PENDING_FILES_PATH, filename)
        
        # Check if the file is already indexed in Milvus (by source name)
        # We query MilvusService directly here, not via the API service
        if milvus_service.get_document_count_by_source(MILVUS_COLLECTION_NAME, filename) > 0:
            logging.info(f"File '{filename}' already exists in Milvus. Moving to processed storage.")
            move_file(file_path, os.path.join(PROCESSED_FILES_PATH, filename))
            skipped_count += 1
            continue

        try:
            # Process single document to get Langchain documents with PKs
            documents = process_single_document(file_path)
            if documents:
                # Add documents to Milvus directly via MilvusService
                inserted_pks = milvus_service.add_documents(MILVUS_COLLECTION_NAME, documents)
                if inserted_pks:
                    logging.info(f"Successfully inserted {len(inserted_pks)} chunks for file '{filename}'.")
                    # Move file to processed storage after successful insertion
                    move_file(file_path, os.path.join(PROCESSED_FILES_PATH, filename))
                    processed_count += 1
                else:
                    logging.error(f"Failed to insert documents for file '{filename}' into Milvus.")
                    failed_count += 1
            else:
                logging.warning(f"No documents extracted from file '{filename}'. Skipping.")
                failed_count += 1 # Count as failed if no documents were extracted
                # Optionally, move such files to a 'failed' directory
                
        except Exception as e:
            logging.error(f"Error processing file '{filename}': {e}", exc_info=True)
            failed_count += 1
            # Optionally, move failed files to a 'failed' directory for manual inspection

    return jsonify({
        'status': 'success',
        'message': f'Document processing completed.',
        'processed_files': processed_count,
        'skipped_files': skipped_count,
        'failed_files': failed_count
    }), 200

@app.route('/api/sources/delete_by_filename', methods=['POST'])
def delete_source_by_filename():
    """
    Deletes documents associated with a specific filename (source) via the Milvus API Service.
    """
    data: dict = request.json
    filename_to_delete = data.get('filename')

    if not filename_to_delete:
        return jsonify({'error': 'Filename is required to delete documents by source.'}), 400

    try:
        response_data = _make_milvus_api_request('post', '/documents/delete_by_source', json_data={'source': filename_to_delete})
        
        # After successful deletion from Milvus, delete the file from processed_storage
        processed_file_path = os.path.join(PROCESSED_FILES_PATH, filename_to_delete)
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
            logging.info(f"Deleted file '{filename_to_delete}' from processed_storage.")
        
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API to delete by filename: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to delete documents by filename via Milvus API: {e}'}), 500

# --- New Endpoints for Milvus Collection Management (via Milvus API Service) ---
@app.route('/api/milvus/collections/list', methods=['GET'])
def list_milvus_collections_api():
    """
    Lists all existing Milvus collections via the Milvus API Service.
    """
    try:
        response_data = _make_milvus_api_request('get', '/collections/list')
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API to list collections: {e}")
        return jsonify({'error': f'Failed to list Milvus collections: {e}'}), 500

@app.route('/api/milvus/collections/describe', methods=['GET'])
def describe_milvus_collection_api():
    """
    Describes a specific Milvus collection via the Milvus API Service.
    Requires 'collection_name' as a query parameter.
    """
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Collection name is required.'}), 400
    try:
        response_data = _make_milvus_api_request('get', '/collections/describe', params={'collection_name': collection_name})
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API to describe collection '{collection_name}': {e}")
        return jsonify({'error': f"Failed to describe collection '{collection_name}': {e}"}), 500

@app.route('/api/milvus/collections/stats', methods=['GET'])
@swag_from({
    'tags': ['Milvus Collections'],
    'parameters': [
        {'name': 'collection_name', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Name of the collection to get stats for'}
    ],
    'responses': {
        200: {'description': 'Collection statistics returned'}
    }
})
def get_milvus_collection_stats_api():
    """
    Gets statistics for a specific Milvus collection via the Milvus API Service.
    Requires 'collection_name' as a query parameter.
    """
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Collection name is required.'}), 400
    try:
        response_data = _make_milvus_api_request('get', '/collections/stats', params={'collection_name': collection_name})
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API to get stats for collection '{collection_name}': {e}")
        return jsonify({'error': f"Failed to get stats for collection '{collection_name}': {e}"}), 500

# NEW: Endpoint to update configuration from .env
@app.route('/api/config/update', methods=['POST'])
@swag_from({
    'tags': ['Configuration'],
    'summary': 'Update configuration variables in the .env file',
    'description': 'Allows dynamic updating of key-value pairs in the .env file. '
                   'Note: Some changes (e.g., MILVUS_URL, EMBEDDING_MODEL) '
                   'may require an application restart to take full effect.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'additionalProperties': {'type': 'string'},
                'example': {
                    "MILVUS_COLLECTION": "new_collection_name",
                    "OPENAI_COMPLETION_MODEL": "gpt-3.5-turbo"
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Configuration updated'},
        400: {'description': 'Invalid request data'},
        500: {'description': 'Error updating .env file'}
    }
})
def update_config():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'Request must be a JSON object containing configuration key-value pairs.'}), 400

    dotenv_path = find_dotenv()
    if not dotenv_path:
        return jsonify({'error': 'Could not find .env file.'}), 500

    updated_keys = []
    restart_required = False
    
    # Variables that require a restart if changed
    # These are variables that are read once at application startup
    # or affect fundamental aspects like server binding or core service initialization.
    restart_sensitive_vars = [
        "SERVER_PORT", # Flask app port binding
        "VECTOR_DIMENSION", # Milvus collection schema (requires recreation)
        "MILVUS_URL", # MilvusService initialization (requires re-instantiation)
        "MILVUS_COLLECTION", # MilvusService initialization (default collection name)
        "OPENAI_EMBEDDING_MODEL", # Embeddings model initialization (requires re-instantiation)
        "SERVER_EMBEDDING_PROVIDER", # Embeddings model initialization
        "MILVUS_API_PORT", # Milvus API service port binding
        "LANGCHAIN_TRACING_V2", # LangChain global settings
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "UPLOAD_FOLDER", # Base paths for file storage
        "PENDING_FILES_PATH",
        "PROCESSED_FILES_PATH"
    ]

    for key, value in data.items():
        # Convert boolean/integer values from string if necessary
        # Note: os.getenv returns string, so we compare string to string
        # For boolean values, ensure they are stored as "true" or "false"
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, int):
            value = str(value)

        # Check if the value actually changed
        current_value = os.getenv(key)
        if current_value != value:
            set_key(dotenv_path, key, value)
            os.environ[key] = value # Update os.environ directly for current process
            updated_keys.append(key)
            logging.info(f"Updated configuration: {key}={value}")

            if key in restart_sensitive_vars:
                restart_required = True

    if updated_keys:
        message = f"Configuration updated for keys: {', '.join(updated_keys)}."
        if restart_required:
            message += " Some changes may require an application restart to take full effect."
        return jsonify({'status': 'success', 'message': message, 'updated_keys': updated_keys, 'restart_required': restart_required})
    else:
        return jsonify({'status': 'success', 'message': 'No configuration changed.', 'updated_keys': [], 'restart_required': False})

@app.route('/api/config/get_dynamic_vars', methods=['GET'])
@swag_from({
    'tags': ['Configuration'],
    'summary': 'Get current values of dynamically changeable configuration variables',
    'description': 'Returns a JSON object containing the current values of configuration variables '
                   'that can be updated via API without requiring a server restart.',
    'responses': {
        200: {
            'description': 'Current dynamic configuration values',
            'schema': {
                'type': 'object',
                'properties': {
                    'OPENAI_API_KEY': {'type': 'string'},
                    'OPENAI_COMPLETION_MODEL': {'type': 'string'},
                    'SEARCH_K_VALUE': {'type': 'integer'},
                    'MILVUS_API_BASE_URL': {'type': 'string'},
                    'MILVUS_API_KEY': {'type': 'string'},
                    'CHUNK_SIZE': {'type': 'integer'},
                    'CHUNK_OVERLAP': {'type': 'integer'},
                    'TESSERACT_CMD_PATH': {'type': 'string'}
                },
                'example': {
                    "OPENAI_API_KEY": "sk-proj-...",
                    "OPENAI_COMPLETION_MODEL": "gpt-4o-mini",
                    "SEARCH_K_VALUE": 4,
                    "MILVUS_API_BASE_URL": "http://127.0.0.1:5000/milvus",
                    "MILVUS_API_KEY": "your_milvus_api_secret_key",
                    "CHUNK_SIZE": 1000,
                    "CHUNK_OVERLAP": 200,
                    "TESSERACT_CMD_PATH": ""
                }
            }
        }
    }
})
def get_dynamic_vars():
    """
    Returns the current values of configuration variables that can be changed dynamically.
    """
    dynamic_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_COMPLETION_MODEL": os.getenv("OPENAI_COMPLETION_MODEL"),
        "SEARCH_K_VALUE": int(os.getenv("SEARCH_K_VALUE", 4)),
        "MILVUS_API_BASE_URL": os.getenv("MILVUS_API_BASE_URL"),
        "MILVUS_API_KEY": os.getenv("MILVUS_API_KEY"),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 200)),
        "TESSERACT_CMD_PATH": os.getenv("TESSERACT_CMD_PATH")
    }
    return jsonify(dynamic_vars)
    
if __name__ == '__main__':
    # Ensure the default collection is created when the main application starts
    # This is where the collection is created if it doesn't exist, with auto_id=False for PK
    VECTOR_DIMENSION_ON_STARTUP = int(os.getenv('VECTOR_DIMENSION', 1536)) # Use a distinct variable for startup
    MILVUS_COLLECTION_NAME_ON_STARTUP = os.getenv('MILVUS_COLLECTION', 'test_data2') # Use a distinct variable for startup
    try:
        milvus_service.create_collection(MILVUS_COLLECTION_NAME_ON_STARTUP, VECTOR_DIMENSION_ON_STARTUP, recreate=False)
        logging.info(f"Milvus collection '{MILVUS_COLLECTION_NAME_ON_STARTUP}' ensured to exist.")
    except Exception as e:
        logging.error(f"Failed to ensure Milvus collection '{MILVUS_COLLECTION_NAME_ON_STARTUP}' exists: {e}")
        # It's critical to exit or handle gracefully if the core service dependency fails at startup
        exit(1)

    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('SERVER_PORT', 5001))) # Read SERVER_PORT dynamically here
