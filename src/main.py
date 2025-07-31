# main.py
import time
import os
import json
import uuid
from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import requests # Add requests library to call API
import logging

load_dotenv('./.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.agent import invoke_agent, stream_agent_response # agent.py still uses MilvusService directly
from src.load_file import load_documents, process_single_document # Import the new helper

app = Flask(__name__)

# Configure URL for Milvus API Service
MILVUS_API_BASE_URL = os.getenv("MILVUS_API_BASE_URL", "http://127.0.0.1:5000/milvus")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY") # Get API Key for Milvus API Service

# --- Helper to make authenticated requests to Milvus API Service ---
def _make_milvus_api_request(method: str, endpoint: str, json_data: dict = None, params: dict = None):
    headers = {}
    if MILVUS_API_KEY:
        headers['X-API-Key'] = MILVUS_API_KEY
    
    url = f"{MILVUS_API_BASE_URL}{endpoint}"
    
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
        chunk (str): Content of the chunk.

    Returns:
        str: Serialized JSON string.
    """
    data = {
        'id': f"emgcmpl-{uuid.uuid4()}",
        'created': int(time.time()),
        'model': os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'),
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
    Handles chat requests, can be streaming or non-streaming.
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
            'sources': [], # Can add sources here if agent returns them
            'id': f'emgcmpl-{uuid.uuid4()}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'),
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
    """
    try:
        response_data = _make_milvus_api_request('get', '/data/show_all')
        sources_data = response_data.get('source_counts', {})
        return jsonify([{'source': source, 'split_count': count} for source, count in sources_data.items()])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API for sources: {e}")
        return jsonify({'error': f'Failed to fetch sources from Milvus API: {e}'}), 500

@app.route('/api/sources/update', methods=['POST'])
def update_source():
    """
    Updates (adds new) documents from the configured storage path to Milvus via the Milvus API Service.
    Only new files (based on source name) will be inserted.
    For very large datasets, consider implementing batching and asynchronous processing.
    """
    storage_path = os.getenv("SERVER_STORAGE_PATH", './storage')
    
    # 1. Load all documents from storage
    all_documents_from_storage = load_documents(storage_path)

    if not all_documents_from_storage:
        return jsonify({'status': 'info', 'message': 'No documents found in storage to process.'})

    # 2. Get existing sources from Milvus
    try:
        existing_sources_response = _make_milvus_api_request('get', '/data/show_all')
        existing_source_counts = existing_sources_response.get('source_counts', {})
        existing_source_names = set(existing_source_counts.keys())
        logging.info(f"Found {len(existing_source_names)} existing sources in Milvus.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching existing sources from Milvus API: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to fetch existing sources from Milvus API: {e}'}), 500

    new_documents_to_insert = []
    skipped_documents_count = 0
    # Use a set to track files processed in this specific batch to avoid internal duplicates
    # if load_documents somehow returns multiple chunks for the same file.
    processed_files_for_this_run = set() 

    # Filter documents: only include those from new files
    for doc in all_documents_from_storage:
        source_name = doc.metadata.get('source')
        if source_name and source_name not in existing_source_names and source_name not in processed_files_for_this_run:
            new_documents_to_insert.append(doc)
            processed_files_for_this_run.add(source_name) # Mark this file as processed for this run
        elif source_name:
            skipped_documents_count += 1
            logging.info(f"Skipping document from existing source: {source_name}")
        else:
            logging.warning(f"Document without a 'source' metadata found. Skipping: {doc.page_content[:50]}...")

    if not new_documents_to_insert:
        return jsonify({'status': 'info', 'message': f'No new documents to add. {skipped_documents_count} documents from existing sources were skipped.'})

    # Convert Langchain Document to JSON format to send via API
    documents_json = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in new_documents_to_insert]

    try:
        # TODO: Implement batching here for very large document lists to avoid single large payload
        # For now, sending all new documents in one go.
        response_data = _make_milvus_api_request('post', '/documents/insert', json_data={'documents': documents_json})
        
        message = f"Successfully inserted {len(new_documents_to_insert)} new documents. {skipped_documents_count} documents from existing sources were skipped."
        response_data['message'] = message # Enhance the response message
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API for update: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to update documents via Milvus API: {e}'}), 500

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
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Milvus API to delete by filename: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to delete documents by filename via Milvus API: {e}'}), 500

@app.route('/api/sources/upload_files', methods=['POST'])
def upload_files_to_milvus():
    """
    Allows users to upload one or more files directly, processes them, and inserts into Milvus.
    Handles duplicate filenames by skipping already existing ones.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'error': 'No selected file'}), 400

    # Get existing sources from Milvus to check for duplicates
    try:
        existing_sources_response = _make_milvus_api_request('get', '/data/show_all')
        existing_source_counts = existing_sources_response.get('source_counts', {})
        existing_source_names = set(existing_source_counts.keys())
        logging.info(f"Found {len(existing_source_names)} existing sources in Milvus for upload check.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching existing sources from Milvus API for upload: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to fetch existing sources from Milvus API for upload: {e}'}), 500

    upload_results = {
        'total_files_received': len(uploaded_files),
        'files_uploaded_successfully': [],
        'files_skipped_duplicates': [],
        'files_failed_processing': []
    }

    temp_dir = './temp_uploads' # Temporary directory for uploaded files
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        filename = file.filename
        if not filename:
            upload_results['files_failed_processing'].append({'filename': 'N/A', 'reason': 'Filename missing'})
            continue

        if filename in existing_source_names:
            upload_results['files_skipped_duplicates'].append(filename)
            logging.info(f"Skipping uploaded file '{filename}' as it already exists in Milvus.")
            continue

        temp_filepath = os.path.join(temp_dir, filename)
        try:
            file.save(temp_filepath) # Save the uploaded file temporarily
            logging.info(f"Saved uploaded file temporarily to: {temp_filepath}")

            # Process the single uploaded file
            processed_docs = process_single_document(temp_filepath)

            if processed_docs:
                # Convert Langchain Document to JSON format
                docs_json = [{'page_content': d.page_content, 'metadata': d.metadata} for d in processed_docs]
                
                # Insert into Milvus
                insert_response = _make_milvus_api_request('post', '/documents/insert', json_data={'documents': docs_json})
                
                if insert_response.get('status') == 'success':
                    upload_results['files_uploaded_successfully'].append(filename)
                    logging.info(f"Successfully processed and inserted uploaded file: {filename}")
                else:
                    upload_results['files_failed_processing'].append({'filename': filename, 'reason': insert_response.get('message', 'Milvus insertion failed')})
                    logging.error(f"Failed to insert documents for uploaded file '{filename}': {insert_response.get('message')}")
            else:
                upload_results['files_failed_processing'].append({'filename': filename, 'reason': 'No documents extracted from file'})
                logging.warning(f"No documents extracted from uploaded file: {filename}")

        except requests.exceptions.RequestException as e:
            upload_results['files_failed_processing'].append({'filename': filename, 'reason': f'Milvus API error: {e}'})
            logging.error(f"Milvus API error during upload processing for '{filename}': {e}")
        except Exception as e:
            upload_results['files_failed_processing'].append({'filename': filename, 'reason': f'Processing error: {e}'})
            logging.error(f"Error processing uploaded file '{filename}': {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                logging.info(f"Cleaned up temporary file: {temp_filepath}")
    
    # Clean up the temporary directory if empty
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return jsonify({'status': 'completed', 'results': upload_results})

# --- New Endpoints for Milvus Collection Management ---
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


if __name__ == '__main__':
    app.debug = True
    port = os.getenv("SERVER_PORT", 3000)
    app.run(threaded=True, port=port, host='127.0.0.1')

