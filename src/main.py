# src/main.py
import os
import uuid
import logging
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv('./.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import các thành phần từ các file khác
from src.database import get_db, AIModel, Collection, AIResponse, RagProgress, create_tables, User, SessionLocal
from src.agent import invoke_agent, stream_agent_response
# THAY ĐỔI: Import hàm xử lý file đã được nâng cấp
from src.load_file import process_and_embed_document
from src.milvus_langchain import milvus_service

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Cấu hình các đường dẫn từ file .env
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", './uploads')

@app.route('/api/ai', methods=['POST'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'example': 'user123'},
                    'name': {'type': 'string', 'example': 'My OpenAI AI'},
                    'provider': {'type': 'string', 'enum': ['openai', 'gemini', 'ollama', 'custom'], 'example': 'openai'},
                    'api_key': {'type': 'string', 'example': 'sk-YOUR_OPENAI_API_KEY'},
                    'embedding_model_name': {'type': 'string', 'example': 'text-embedding-3-small'},
                    'chat_model_name': {'type': 'string', 'example': 'gpt-3.5-turbo'},
                    'embedding_dim': {'type': 'integer', 'example': 1536},
                    'tool': {'type': 'string', 'example': 'proxy-n8n'},
                    'ai_domain': {'type': 'string', 'example': 'https://your-proxy-endpoint.com'}
                },
                'required': ['user_id', 'name', 'provider', 'embedding_model_name', 'chat_model_name']
            }
        }
    ],
    'responses': {
        201: {'description': 'AI Model created successfully'},
        400: {'description': 'Missing required fields'},
        409: {'description': 'AI Model with this name already exists for user'}
    }
})
def create_ai():
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    provider = data.get('provider')
    api_key = data.get('api_key') # Có thể là None nếu provider là ollama
    embedding_model_name = data.get('embedding_model_name')
    chat_model_name = data.get('chat_model_name')
    embedding_dim = data.get('embedding_dim')
    tool = data.get('tool')
    ai_domain = data.get('ai_domain')

    if not all([user_id, name, provider, embedding_model_name, chat_model_name, embedding_dim]):
        return jsonify({'error': 'Missing required fields: user_id, name, provider, embedding_model_name, chat_model_name, embedding_dim'}), 400

    if provider in ['openai', 'gemini'] and not api_key:
        return jsonify({'error': f'API key is required for provider {provider}'}), 400


    db = next(get_db())
    try:
        existing_ai = db.query(AIModel).filter_by(user_id=user_id, name=name).first()
        if existing_ai:
            return jsonify({'error': 'AI Model with this name already exists for user'}), 409

        new_ai = AIModel(
            user_id=user_id, name=name, provider=provider, api_key=api_key,
            tool=tool,
            ai_domain=ai_domain,
            embedding_model_name=embedding_model_name,
            chat_model_name=chat_model_name,
            embedding_dim=embedding_dim
        )
        db.add(new_ai)
        db.commit()
        db.refresh(new_ai)
        return jsonify({
            'message': 'AI Model created successfully',
            'ai_id': str(new_ai.id),
            'name': new_ai.name
        }), 201
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating AI model: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create AI model'}), 500
    finally:
        db.close()


@app.route('/api/ai/<uuid:ai_id>', methods=['PUT'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'example': 'My Edited AI'},
                    'provider': {'type': 'string', 'enum': ['openai', 'gemini', 'ollama', 'custom']},
                    'api_key': {'type': 'string', 'example': 'sk-NEW_API_KEY'},
                    'embedding_model_name': {'type': 'string', 'example': 'text-embedding-3-large'},
                    'chat_model_name': {'type': 'string', 'example': 'gpt-4'},
                    'embedding_dim': {'type': 'integer', 'example': 3072},
                    'tool': {'type': 'string', 'example': 'retrieval'},
                    'ai_domain': {'type': 'string', 'example': 'legaltech'},

                }
            }
        }
    ],
    'responses': {
        200: {'description': 'AI Model updated successfully'},
        400: {'description': 'Missing required fields'},
        404: {'description': 'AI Model not found or not owned by user'},
        409: {'description': 'Another AI Model with this name already exists for user'}
    }
})
def edit_ai(ai_id):
    data = request.json
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
    if not ai_info:
        db.close()
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404

    # Check for name conflict
    new_name = data.get('name')
    if new_name and new_name != ai_info.name:
        existing = db.query(AIModel).filter(AIModel.user_id == user_id, AIModel.name == new_name).first()
        if existing:
            db.close()
            return jsonify({'error': 'Another AI Model with this name already exists for user'}), 409

    # Cập nhật các trường cho phép
    updatable_fields = ['name', 'provider', 'api_key', 'embedding_model_name', 'chat_model_name', 'embedding_dim', 'tool', 'ai_domain']
    for field in updatable_fields:
        if field in data:
            setattr(ai_info, field, data[field])

    # Bắt buộc API key nếu đổi sang openai hoặc gemini mà không có key
    if ai_info.provider in ['openai', 'gemini'] and not ai_info.api_key:
        db.close()
        return jsonify({'error': f'API key is required for provider {ai_info.provider}'}), 400

    try:
        db.commit()
        db.refresh(ai_info)
        return jsonify({'message': 'AI Model updated successfully'}), 200
    except Exception as e:
        db.rollback()
        logging.error(f"Error updating AI model: {e}", exc_info=True)
        return jsonify({'error': 'Failed to update AI model'}), 500
    finally:
        db.close()


@app.route('/api/ai/<uuid:ai_id>', methods=['GET'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'AI Model details'}, 404: {'description': 'AI Model not found'}}
})
def get_ai(ai_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
    db.close()

    if ai_info:
        return jsonify({
            'id': str(ai_info.id), 'name': ai_info.name, 'provider': ai_info.provider,
            'tool':ai_info.tool,
            'ai_domain': ai_info.ai_domain,
            'embedding_model_name': ai_info.embedding_model_name,
            'chat_model_name': ai_info.chat_model_name,
            'embedding_dim': ai_info.embedding_dim,
            'created_at': ai_info.created_at.isoformat()
        }), 200
    return jsonify({'error': 'AI Model not found or not owned by user'}), 404

@app.route('/api/ai/<uuid:ai_id>', methods=['DELETE'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'AI Model deleted'}, 404: {'description': 'AI Model not found'}}
})
def delete_ai(ai_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        ai_to_delete = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if ai_to_delete:
            db.delete(ai_to_delete)
            db.commit()
            return jsonify({'message': f'AI Model {ai_id} deleted successfully'}), 200
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting AI model {ai_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete AI model'}), 500
    finally:
        db.close()


# --- Collection Endpoints ---

@app.route('/api/collections', methods=['POST'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'ai_id': {'type': 'string', 'format': 'uuid'},
                'user_id': {'type': 'string'},
                'name': {'type': 'string'}
            },
            'required': ['ai_id', 'user_id', 'name']
        }
    }],
    'responses': {201: {'description': 'Collection created'}, 404: {'description': 'AI Model not found'}, 409: {'description': 'Collection name exists'}}
})
def create_collection():
    data = request.json
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    name = data.get('name')

    if not all([ai_id, user_id, name]):
        return jsonify({'error': 'Missing required fields'}), 400

    db = next(get_db())
    try:
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'AI Model not found or not owned by user'}), 404

        existing_collection = db.query(Collection).filter_by(ai_id=ai_id, name=name).first()
        if existing_collection:
            return jsonify({'error': 'Collection with this name already exists for this AI'}), 409

        milvus_collection_name = f"col_{str(ai_id).replace('-', '_')}_{str(uuid.uuid4()).replace('-', '_')[:8]}"
        milvus_service.create_collection(milvus_collection_name, ai_info.embedding_dim)

        new_collection = Collection(ai_id=ai_id, name=name, milvus_collection_name=milvus_collection_name)
        db.add(new_collection)
        db.commit()
        db.refresh(new_collection)
        return jsonify({
            'message': 'Collection created successfully',
            'collection_id': str(new_collection.id),
            'name': new_collection.name,
            'milvus_collection_name': new_collection.milvus_collection_name
        }), 201
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating collection: {e}", exc_info=True)
        # Cố gắng xóa collection trên milvus nếu đã tạo
        if 'milvus_collection_name' in locals():
            milvus_service.drop_collection(milvus_collection_name)
        return jsonify({'error': 'Failed to create collection'}), 500
    finally:
        db.close()

@app.route('/api/ai/<uuid:ai_id>/collections', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the AI Model.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'List of collections for the AI Model'},
        404: {'description': 'AI Model not found or not owned by user'}
    }
})
def list_collections_for_ai(ai_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'AI Model not found or not owned by user'}), 404

        collections = db.query(Collection).filter(Collection.ai_id == ai_id).all()
        
        collection_list = [{
            'id': str(c.id),
            'name': c.name,
            'milvus_collection_name': c.milvus_collection_name,
            'created_at': c.created_at.isoformat()
        } for c in collections]

        return jsonify({'collections': collection_list}), 200
    finally:
        db.close()

@app.route('/api/collections/<uuid:collection_id>', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'Collection details'}, 404: {'description': 'Collection not found'}}
})
def get_collection(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).join(AIModel).filter(
            Collection.id == collection_id,
            AIModel.user_id == user_id
        ).first()

        if collection_info:
            return jsonify({
                'id': str(collection_info.id), 'ai_id': str(collection_info.ai_id),
                'name': collection_info.name,
                'milvus_collection_name': collection_info.milvus_collection_name,
                'created_at': collection_info.created_at.isoformat()
            }), 200
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    finally:
        db.close()

@app.route('/api/collections/<uuid:collection_id>', methods=['DELETE'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'Collection deleted'}, 404: {'description': 'Collection not found'}}
})
def delete_collection(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_to_delete = db.query(Collection).join(AIModel).filter(
            Collection.id == collection_id,
            AIModel.user_id == user_id
        ).first()

        if collection_to_delete:
            milvus_service.drop_collection(collection_to_delete.milvus_collection_name)
            db.delete(collection_to_delete)
            db.commit()
            return jsonify({'message': f'Collection {collection_id} deleted successfully'}), 200
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete collection'}), 500
    finally:
        db.close()

# --- Document & Chat Endpoints ---

@app.route('/api/documents/upload', methods=['POST'])
@swag_from({
    'tags': ['Documents'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {'name': 'file', 'in': 'formData', 'type': 'file', 'required': True, 'description': 'The document to upload (PDF, MD, JSON).'},
        {'name': 'ai_id', 'in': 'formData', 'type': 'string', 'required': True},
        {'name': 'collection_id', 'in': 'formData', 'type': 'string', 'required': True},
        {'name': 'user_id', 'in': 'formData', 'type': 'string', 'required': True}
    ],
    'responses': {
        200: {'description': 'Document processed and added to collection'},
        400: {'description': 'Bad request (missing fields, invalid file type)'},
        404: {'description': 'AI Model or Collection not found'},
        500: {'description': 'Failed to process document'}
    }
})
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    ai_id = request.form.get('ai_id')
    collection_id = request.form.get('collection_id')
    user_id = request.form.get('user_id')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not all([ai_id, collection_id, user_id]):
        return jsonify({'error': 'Missing required fields: ai_id, collection_id, user_id'}), 400

    db = next(get_db())
    temp_filepath = None
    try:
        # Xác thực thông tin và quyền sở hữu
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'AI Model not found or not owned by user'}), 404
        
        collection_info = db.query(Collection).filter(Collection.id == collection_id, Collection.ai_id == ai_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found or does not belong to the specified AI model'}), 404

        filename = secure_filename(file.filename)
        
        # Lưu file vào một thư mục tạm thời để xử lý
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        temp_filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(temp_filepath)
        logging.info(f"File saved temporarily to {temp_filepath}")

        # Gọi hàm xử lý file (load, chunk, embed, insert to Milvus)
        success = process_and_embed_document(
            file_path=temp_filepath,
            collection_info=collection_info,
            ai_info=ai_info
        )
        
        if success:
            return jsonify({'message': f'Document "{filename}" processed and added to collection successfully'}), 200
        else:
            return jsonify({'error': f'Failed to process document "{filename}"'}), 500

    except Exception as e:
        logging.error(f"Error during document upload: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during file upload.'}), 500
    finally:
        # Dọn dẹp file tạm sau khi xử lý xong
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logging.info(f"Temporary file {temp_filepath} removed.")
        db.close()


@app.route('/api/chat/completions', methods=['POST'])
@swag_from({
    'tags': ['Chat'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'messages': {'type': 'array', 'items': {'type': 'object', 'properties': {'role': {'type': 'string'}, 'content': {'type': 'string'}}}},
                'ai_id': {'type': 'string', 'format': 'uuid'},
                'user_id': {'type': 'string'},
                'collection_id': {'type': 'string', 'format': 'uuid'},
                'stream': {'type': 'boolean', 'default': False}
            },
            'required': ['messages', 'ai_id', 'user_id', 'collection_id']
        }
    }],
    'responses': {200: {'description': 'Chat completion successful'}, 500: {'description': 'Internal Server Error'}}
})
def chat():
    data = request.json
    messages = data.get('messages')
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    collection_id = data.get('collection_id')
    stream = data.get('stream', False)

    if not all([messages, ai_id, user_id, collection_id]):
        return jsonify({'error': 'Missing required fields'}), 400

    db = next(get_db())
    try:
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
       

        if not ai_info:
            return jsonify({'error': 'AI Model not found or not owned by user'}), 404


        tool = ai_info.tool or "default"

        # === Proxy Tools ===
        if tool == "proxy-ai":
            result, status = handle_proxy_ai(ai_info, messages)
            return jsonify(result), status

        elif tool == "proxy-n8n":
            result, status = handle_proxy_n8n(ai_info, messages)
            return jsonify(result), status

        collection_info = db.query(Collection).filter(Collection.id == collection_id, Collection.ai_id == ai_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found or not owned by user'}), 404

        # === Default flow ===
        question = messages[-1]['content'] if messages else ""
        milvus_collection_name = collection_info.milvus_collection_name

        if stream:
            return Response(stream_agent_response(ai_info, question, milvus_collection_name), mimetype='text/event-stream')
        else:
            response_data = invoke_agent(ai_info, question, milvus_collection_name)
            # Chuyển đổi Document objects thành dicts trước khi trả về JSON
            if 'sources' in response_data and response_data['sources']:
                 response_data['sources'] = [
                    {'page_content': doc.page_content, 'metadata': doc.metadata}
                    for doc in response_data['sources']
                ]
            return jsonify(response_data)

    except Exception as e:
        logging.error(f"An unexpected error occurred during chat: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during chat.'}), 500
    finally:
        db.close()

# --- Quản lý nguồn dữ liệu trong Collection ---

@app.route('/api/collections/<uuid:collection_id>/sources', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'List of unique source filenames'}, 404: {'description': 'Collection not found'}}
})
def get_collection_sources(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).join(AIModel).filter(
            Collection.id == collection_id,
            AIModel.user_id == user_id
        ).first()
        if collection_info:
            sources = milvus_service.get_all_sources_in_collection(collection_info.milvus_collection_name)
            return jsonify({'sources': sources}), 200
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    finally:
        db.close()

@app.route('/api/collections/<uuid:collection_id>/sources/<path:source_filename>', methods=['DELETE'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True},
        {'name': 'source_filename', 'in': 'path', 'type': 'string', 'required': True},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'Source documents deleted'}, 404: {'description': 'Collection or Source not found'}}
})
def delete_source_documents(collection_id, source_filename):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).join(AIModel).filter(
            Collection.id == collection_id,
            AIModel.user_id == user_id
        ).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found or not owned by user'}), 404
        
        deleted_count = milvus_service.delete_documents_by_source(collection_info.milvus_collection_name, source_filename)
        
        if deleted_count > 0:
            return jsonify({'message': f'Deleted {deleted_count} documents for source {source_filename}'}), 200
        else:
            return jsonify({'message': f'No documents found for source {source_filename}'}), 404
    except Exception as e:
        logging.error(f"Error deleting source {source_filename}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete source documents'}), 500
    finally:
        db.close()

@app.route('/api/document-schema', methods=['GET'])
def get_document_schema():
    return {
        "schema": {
            "vector": "float[] (length = embedding_dim)",
            "text": "string",
            "metadata": {
                "source": "string",
                "page_number": "integer",
                "chunk_index": "integer",
                "created_at": "datetime"
            }
        },
        "example": {
            "vector": [0.123, 0.456, "..."],
            "text": "This is an example chunk of text...",
            "metadata": {
                "source": "contract_v1.pdf",
                "page_number": 10,
                "chunk_index": 5,
                "created_at": "2025-08-04T12:00:00Z"
            }
        }
    }

@app.route('/api/documents', methods=['GET'])
@swag_from({
    'tags': ['Documents'],
    'parameters': [
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True},
        {'name': 'ai_id', 'in': 'query', 'type': 'string', 'format': 'uuid', 'required': False},
        {'name': 'collection_id', 'in': 'query', 'type': 'string', 'format': 'uuid', 'required': False},
        {'name': 'source_filename', 'in': 'query', 'type': 'string', 'required': False}
    ],
    'responses': {200: {'description': 'List of matching documents'}, 400: {'description': 'Invalid request'}}
})
def get_documents():
    user_id = request.args.get('user_id')
    ai_id = request.args.get('ai_id')
    collection_id = request.args.get('collection_id')
    source_filename = request.args.get('source_filename')

    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())

    try:
        collections = []

        collections = []

        if collection_id:
            collection = db.query(Collection).join(AIModel).filter(
                Collection.id == collection_id,
                AIModel.user_id == user_id
            ).first()
            if not collection:
                return jsonify({'error': 'Collection not found or not owned by user'}), 404
            collections.append(collection)

        elif ai_id:
            ai = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
            if not ai:
                return jsonify({'error': 'AI Model not found or not owned by user'}), 404
            collections = ai.collections

        else:
            # Lấy tất cả collection thuộc user
            collections = db.query(Collection).join(AIModel).filter(
                AIModel.user_id == user_id
            ).all()


        all_chunks = []

        for col in collections:
            if source_filename:
                chunks = milvus_service.get_chunks_by_source(col.milvus_collection_name, source_filename)
            else:
                chunks = milvus_service.get_all_chunks(col.milvus_collection_name)
            all_chunks.extend(chunks)

        return jsonify({'documents': all_chunks}), 200

    except Exception as e:
        logging.error(f"Error getting documents: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get documents'}), 500
    finally:
        db.close()
        

@app.route('/api/users', methods=['GET'])
def get_user():
    user_id = request.args.get('user_id')
    email = request.args.get('email')

    if not user_id and not email:
        return jsonify({"error": "Vui lòng cung cấp user_id hoặc email"}), 400

    db = SessionLocal()
    try:
        query = db.query(User)
        if user_id:
            user = query.filter(User.id == user_id).first()
        else:
            user = query.filter(User.email == email).first()

        if not user:
            return jsonify({"error": "User không tồn tại"}), 404

        return jsonify({
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat(),
            "ai_models": [
                {
                    "id": str(ai.id),
                    "name": ai.name,
                    "provider": ai.provider,
                    "embedding_model_name": ai.embedding_model_name,
                    "chat_model_name": ai.chat_model_name,
                    "embedding_dim": ai.embedding_dim,
                    "tool": ai.tool,
                    "ai_domain": ai.ai_domain
                }
                for ai in user.ai_models
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

from werkzeug.security import generate_password_hash
from sqlalchemy.exc import IntegrityError

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    full_name = data.get("full_name")

    if not email or not password:
        return jsonify({"error": "Email và password là bắt buộc"}), 400

    db = SessionLocal()
    try:
        hashed_password = generate_password_hash(password)
        new_user = User(
            email=email,
            password_hash=hashed_password,
            full_name=full_name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return jsonify({
            "id": str(new_user.id),
            "email": new_user.email,
            "full_name": new_user.full_name,
            "created_at": new_user.created_at.isoformat()
        }), 201

    except IntegrityError:
        db.rollback()
        return jsonify({"error": "Email đã tồn tại"}), 409
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()



from werkzeug.security import check_password_hash
import jwt
import datetime

SECRET_KEY = os.getenv("SECRET_KEY", "EMG")

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email và password là bắt buộc"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Sai email hoặc mật khẩu"}), 401

        # Tạo JWT token
        token = jwt.encode({
            "user_id": str(user.id),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
        }, SECRET_KEY, algorithm="HS256")

        return jsonify({
            "access_token": token,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/user-id', methods=['GET'])
def get_user_id_from_email():
    email = request.args.get('email')

    if not email:
        return jsonify({"error": "Email là bắt buộc"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404

        return jsonify({"user_id": str(user.id)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# from pymilvus import utility
# print(utility.list_collections())
def handle_proxy_ai(ai_info, messages):
    import requests
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(
            ai_info.ai_domain,
            json={"messages": messages},
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        return response.json(), 200
    except requests.RequestException as e:
        logging.error(f"[Proxy AI] Error: {e}", exc_info=True)
        return {'error': 'Failed to connect to proxy AI'}, 502


def handle_proxy_n8n(ai_info, messages):
    import requests
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(
            ai_info.ai_domain,
            json={"messages": messages},
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        return response.json(), 200
    except requests.RequestException as e:
        logging.error(f"[Proxy N8N] Error: {e}", exc_info=True)
        return {'error': 'Failed to connect to proxy N8N'}, 502


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    create_tables()
    app.run(debug=True, host='0.0.0.0', port=3030)

