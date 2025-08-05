# src/main.py
import os
import uuid
import time
import json
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
from flask_cors import CORS

from dotenv import load_dotenv, set_key, find_dotenv
load_dotenv('./.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.database import get_db, AIModel, Collection, AIResponse, RagProgress, create_tables # Đã sửa AILog thành AIResponse
from src.agent import invoke_agent, stream_agent_response
from src.load_file import process_single_document, move_file, get_embedding_model
from src.milvus_langchain import milvus_service

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", './uploads')
PENDING_FILES_PATH = os.getenv("PENDING_FILES_PATH", './pending_files')
PROCESSED_FILES_PATH = os.getenv("PROCESSED_FILES_PATH", './processed_storage')
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "my_rag_collection")


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
                    'provider': {'type': 'string', 'enum': ['openai', 'gemini', 'custom'], 'example': 'openai'},
                    'api_key': {'type': 'string', 'example': 'sk-YOUR_OPENAI_API_KEY'},
                    'embedding_model_name': {'type': 'string', 'example': 'text-embedding-ada-002'}, # Tên model embedding
                    'chat_model_name': {'type': 'string', 'example': 'gpt-3.5-turbo'},               # Tên model chat
                    'embedding_dim': {'type': 'integer', 'example': 1536}
                },
                'required': ['user_id', 'name', 'provider', 'api_key', 'embedding_model_name', 'chat_model_name']
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
    api_key = data.get('api_key')
    
    # Lấy tên mô hình embedding và chat riêng biệt
    embedding_model_name = data.get('embedding_model_name')
    chat_model_name = data.get('chat_model_name')
    
    embedding_dim = data.get('embedding_dim', 1536) # Mặc định cho OpenAI ada-002

    # Kiểm tra các trường mới
    if not all([user_id, name, provider, api_key, embedding_model_name, chat_model_name]):
        return jsonify({'error': 'Missing required fields: user_id, name, provider, api_key, embedding_model_name, chat_model_name'}), 400

    db = next(get_db())
    try:
        existing_ai = db.query(AIModel).filter_by(user_id=user_id, name=name).first()
        if existing_ai:
            return jsonify({'error': 'AI Model with this name already exists for user'}), 409

        new_ai = AIModel(
            user_id=user_id,
            name=name,
            provider=provider,
            api_key=api_key,
            
            # Gán các trường mới
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
            'name': new_ai.name,
            'provider': new_ai.provider,
            'embedding_model_name': new_ai.embedding_model_name,
            'chat_model_name': new_ai.chat_model_name,
            'embedding_dim': new_ai.embedding_dim
        }), 201
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating AI model: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create AI model'}), 500
    finally:
        db.close()


@app.route('/api/documents/upload', methods=['POST'])
@swag_from({
    'tags': ['Documents'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'The PDF document to upload.'
        },
        {
            'name': 'ai_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'ID of the AI model to use for embedding.'
        },
        {
            'name': 'collection_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'ID of the collection to add documents to.'
        },
        {
            'name': 'user_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'ID of the user uploading the document.'
        }
    ],
    'responses': {
        200: {'description': 'Document processed and added to Milvus'},
        400: {'description': 'No file part, Invalid file type, or Missing required fields'},
        404: {'description': 'AI Model or Collection not found'},
        500: {'description': 'Failed to process document'}
    }
})
def upload_document():
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    ai_id = request.form.get('ai_id')
    collection_id = request.form.get('collection_id')
    user_id = request.form.get('user_id')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not all([ai_id, collection_id, user_id]):
        return jsonify({'error': 'Missing required fields: ai_id, collection_id, user_id'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
    collection_info = db.query(Collection).filter(Collection.id == collection_id, Collection.ai_id == ai_id).first() # Kiểm tra ai_id cho collection
    db.close()

    if not ai_info:
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404
    if not collection_info:
        return jsonify({'error': 'Collection not found or not owned by user'}), 404 # Collection thuộc về ai_id, không phải user_id trực tiếp

    milvus_collection_name = collection_info.milvus_collection_name

    # Xử lý file PDF
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        pending_filepath = os.path.join(PENDING_FILES_PATH, filename)
        
        os.makedirs(PENDING_FILES_PATH, exist_ok=True)
        os.makedirs(PROCESSED_FILES_PATH, exist_ok=True)
        
        file.save(pending_filepath)
        logging.info(f"Đang xử lý tệp PDF: {filename}")

        try:
            # Truyền ai_info vào process_single_document
            success = process_single_document(
                file_path=pending_filepath,
                # ai_info=ai_info, # Truyền cả đối tượng ai_info
                # collection_name=milvus_collection_name
            )

            if success:
                # Di chuyển file đã xử lý thành công
                move_file(filename, PROCESSED_FILES_PATH)
                return jsonify({'message': f'Document {filename} processed and added to Milvus'}), 200
            else:
                return jsonify({'error': f'Failed to process document {filename}'}), 500
        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}", exc_info=True)
            # Dọn dẹp file bị lỗi nếu cần, hoặc di chuyển sang thư mục lỗi
            if os.path.exists(pending_filepath):
                os.remove(pending_filepath)
            return jsonify({'error': f'Failed to process document {filename}: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400


@app.route('/api/chat/completions', methods=['POST'])
@swag_from({
    'tags': ['Chat'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'messages': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'role': {'type': 'string', 'example': 'user'},
                                'content': {'type': 'string', 'example': 'What is the price of product X?'}
                            }
                        }
                    },
                    'ai_id': {'type': 'string', 'format': 'uuid', 'example': 'a1b2c3d4-e5f6-7890-1234-567890abcdef'},
                    'user_id': {'type': 'string', 'example': 'user123'},
                    'collection_id': {'type': 'string', 'format': 'uuid', 'example': 'f1e2d3c4-b5a6-9876-5432-10fedcba9876'},
                    'stream': {'type': 'boolean', 'default': False, 'example': False}
                },
                'required': ['messages', 'ai_id', 'user_id', 'collection_id']
            }
        }
    ],
    'responses': {
        200: {'description': 'Successful chat completion', 'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}, 'sources': {'type': 'array', 'items': {'type': 'object'}}}}},
        500: {'description': 'Internal Server Error'}
    }
})
def chat():
    data: dict = request.json
    
    messages = data.get('messages')
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    collection_id = data.get('collection_id')
    stream = data.get('stream', False)

    if not all([messages, ai_id, user_id, collection_id]):
        return jsonify({'error': 'Missing required fields: messages, ai_id, user_id, collection_id'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
    collection_info = db.query(Collection).filter(Collection.id == collection_id, Collection.ai_id == ai_id).first() # Kiểm tra ai_id cho collection

    if not ai_info:
        db.close()
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404
    if not collection_info:
        db.close()
        return jsonify({'error': 'Collection not found or not owned by user'}), 404

    question = messages[-1]['content'] if messages else ""
    milvus_collection_name = collection_info.milvus_collection_name

    try:
        if stream:
            return Response(stream_agent_response(ai_info, question, milvus_collection_name), mimetype='text/event-stream')
        else:
            response_data = invoke_agent(ai_info, question, milvus_collection_name)
            answer_content = response_data.get('answer', '')
            sources = response_data.get('sources', [])
            
            # log_to_db đã được gọi bên trong invoke_agent/stream_agent_response
            
            return jsonify({'answer': answer_content, 'sources': sources})

    except ValueError as e:
        logging.error(f"Error invoking LLM for AI {ai_id}: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during chat: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during chat.'}), 500
    finally:
        db.close()


@app.route('/api/ai/<uuid:ai_id>', methods=['GET'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the AI Model.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'AI Model details'},
        404: {'description': 'AI Model not found'}
    }
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
            'id': str(ai_info.id),
            'name': ai_info.name,
            'provider': ai_info.provider,
            'embedding_model_name': ai_info.embedding_model_name, # Trả về tên model embedding
            'chat_model_name': ai_info.chat_model_name,           # Trả về tên model chat
            'embedding_dim': ai_info.embedding_dim,
            'created_at': ai_info.created_at.isoformat()
        }), 200
    return jsonify({'error': 'AI Model not found or not owned by user'}), 404


@app.route('/api/ai/<uuid:ai_id>', methods=['DELETE'])
@swag_from({
    'tags': ['AI Models'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the AI Model to delete.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'AI Model deleted successfully'},
        404: {'description': 'AI Model not found'},
        500: {'description': 'Failed to delete AI Model'}
    }
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
        return jsonify({'error': f'Failed to delete AI model {ai_id}'}), 500
    finally:
        db.close()


@app.route('/api/collections', methods=['POST'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'ai_id': {'type': 'string', 'format': 'uuid', 'example': 'a1b2c3d4-e5f6-7890-1234-567890abcdef'},
                    'user_id': {'type': 'string', 'example': 'user123'},
                    'name': {'type': 'string', 'example': 'My PDF Collection'}
                },
                'required': ['ai_id', 'user_id', 'name']
            }
        }
    ],
    'responses': {
        201: {'description': 'Collection created successfully'},
        400: {'description': 'Missing required fields'},
        404: {'description': 'AI Model not found'},
        409: {'description': 'Collection with this name already exists for AI'}
    }
})
def create_collection():
    data = request.json
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    name = data.get('name')

    if not all([ai_id, user_id, name]):
        return jsonify({'error': 'Missing required fields: ai_id, user_id, name'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
    if not ai_info:
        db.close()
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404

    try:
        existing_collection = db.query(Collection).filter_by(ai_id=ai_id, name=name).first()
        if existing_collection:
            return jsonify({'error': 'Collection with this name already exists for this AI'}), 409

        # Tạo tên collection Milvus duy nhất, có thể kết hợp ai_id và hash của name
        milvus_collection_name = f"col_{str(ai_id).replace('-', '_')}_{str(uuid.uuid4()).replace('-', '_')[:8]}"
        
        # Tạo Milvus collection
        milvus_service.create_collection(milvus_collection_name, ai_info.embedding_dim)

        new_collection = Collection(
            ai_id=ai_id,
            name=name,
            milvus_collection_name=milvus_collection_name
        )
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
        return jsonify({'error': 'Failed to create collection'}), 500
    finally:
        db.close()


@app.route('/api/collections/<uuid:collection_id>', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the Collection.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'Collection details'},
        404: {'description': 'Collection not found'}
    }
})
def get_collection(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
    
    if collection_info:
        # Kiểm tra quyền sở hữu bằng cách join hoặc kiểm tra riêng AIModel
        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        db.close()
        if ai_info:
            return jsonify({
                'id': str(collection_info.id),
                'ai_id': str(collection_info.ai_id),
                'name': collection_info.name,
                'milvus_collection_name': collection_info.milvus_collection_name,
                'created_at': collection_info.created_at.isoformat()
            }), 200
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    db.close()
    return jsonify({'error': 'Collection not found'}), 404


@app.route('/api/collections/<uuid:collection_id>', methods=['DELETE'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the Collection to delete.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'Collection deleted successfully'},
        404: {'description': 'Collection not found'},
        500: {'description': 'Failed to delete Collection'}
    }
})
def delete_collection(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_to_delete = db.query(Collection).filter(Collection.id == collection_id).first()
        if collection_to_delete:
            # Kiểm tra quyền sở hữu
            ai_info = db.query(AIModel).filter(AIModel.id == collection_to_delete.ai_id, AIModel.user_id == user_id).first()
            if not ai_info:
                return jsonify({'error': 'Collection not found or not owned by user'}), 404

            # Xóa Milvus collection liên quan
            milvus_service.drop_collection(collection_to_delete.milvus_collection_name)
            
            db.delete(collection_to_delete)
            db.commit()
            return jsonify({'message': f'Collection {collection_id} deleted successfully'}), 200
        return jsonify({'error': 'Collection not found'}), 404
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to delete collection {collection_id}'}), 500
    finally:
        db.close()


@app.route('/api/collections/<uuid:collection_id>/sources', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the Collection.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'List of unique source filenames in the collection'},
        404: {'description': 'Collection not found'}
    }
})
def get_collection_sources(collection_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    collection_info = db.query(Collection).filter(Collection.id == collection_id).first()

    if collection_info:
        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        db.close()
        if ai_info:
            sources = milvus_service.get_all_sources_in_collection(collection_info.milvus_collection_name)
            return jsonify({'sources': sources}), 200
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    db.close()
    return jsonify({'error': 'Collection not found'}), 404


@app.route('/api/collections/<uuid:collection_id>/sources/<path:source_filename>/chunks', methods=['GET'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the Collection.'},
        {'name': 'source_filename', 'in': 'path', 'type': 'string', 'required': True, 'description': 'Filename of the source document.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'List of chunks for the specified source'},
        404: {'description': 'Collection or Source not found'}
    }
})
def get_chunks_from_source(collection_id, source_filename):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    collection_info = db.query(Collection).filter(Collection.id == collection_id).first()

    if collection_info:
        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        db.close()
        if ai_info:
            chunks = milvus_service.get_chunks_by_source(collection_info.milvus_collection_name, source_filename)
            if chunks:
                return jsonify({'chunks': chunks}), 200
            return jsonify({'message': 'No chunks found for this source or source does not exist.'}), 404
        return jsonify({'error': 'Collection not found or not owned by user'}), 404
    db.close()
    return jsonify({'error': 'Collection not found'}), 404


@app.route('/api/collections/<uuid:collection_id>/sources/<path:source_filename>', methods=['DELETE'])
@swag_from({
    'tags': ['Collections'],
    'parameters': [
        {'name': 'collection_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the Collection.'},
        {'name': 'source_filename', 'in': 'path', 'type': 'string', 'required': True, 'description': 'Filename of the source document to delete.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'Source documents deleted successfully'},
        404: {'description': 'Collection or Source not found'},
        500: {'description': 'Failed to delete source documents'}
    }
})
def delete_source_documents(collection_id, source_filename):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if collection_info:
            ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
            if not ai_info:
                return jsonify({'error': 'Collection not found or not owned by user'}), 404
            
            # Xóa tài liệu khỏi Milvus
            deleted_count = milvus_service.delete_documents_by_source(collection_info.milvus_collection_name, source_filename)
            
            if deleted_count > 0:
                # Optionally, delete the physical file if it exists in PROCESSED_FILES_PATH
                processed_filepath = os.path.join(PROCESSED_FILES_PATH, source_filename)
                if os.path.exists(processed_filepath):
                    os.remove(processed_filepath)
                    logging.info(f"Deleted physical file: {source_filename}")
                return jsonify({'message': f'Deleted {deleted_count} documents for source {source_filename}'}), 200
            else:
                return jsonify({'message': f'No documents found for source {source_filename} in collection {collection_id}'}), 404
        return jsonify({'error': 'Collection not found'}), 404
    except Exception as e:
        logging.error(f"Error deleting source {source_filename} from collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to delete source documents: {str(e)}'}), 500
    finally:
        db.close()


@app.route('/api/log/ai/<uuid:ai_id>', methods=['GET'])
@swag_from({
    'tags': ['Logging'],
    'parameters': [
        {'name': 'ai_id', 'in': 'path', 'type': 'string', 'format': 'uuid', 'required': True, 'description': 'ID of the AI Model.'},
        {'name': 'user_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'ID of the user.'}
    ],
    'responses': {
        200: {'description': 'List of AI responses for the specified AI model'},
        404: {'description': 'AI Model not found'}
    }
})
def get_ai_responses(ai_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    db = next(get_db())
    ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()

    if not ai_info:
        db.close()
        return jsonify({'error': 'AI Model not found or not owned by user'}), 404
    
    responses = db.query(AIResponse).filter_by(ai_id=ai_id).order_by(AIResponse.create_at.desc()).all() # Đã sửa AILog thành AIResponse
    
    response_list = []
    for res in responses:
        # Lấy các bản ghi RagProgress liên quan
        rag_progresses = db.query(RagProgress).filter_by(question_id=res.id).all()
        sources_info = [
            {"page_content": rag.ann_return, "metadata": rag.rag_metadata}
            for rag in rag_progresses
        ]

        response_list.append({
            'id': res.id,
            'question': res.question,
            'answer': res.ai_answer,
            'sources': sources_info, # Thêm thông tin nguồn
            'answered_at': res.answertime.isoformat(),
            'logged_at': res.create_at.isoformat()
        })
    db.close()
    return jsonify({'responses': response_list}), 200


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PENDING_FILES_PATH, exist_ok=True)
    os.makedirs(PROCESSED_FILES_PATH, exist_ok=True)
    
    create_tables() # Tạo các bảng database
    app.run(debug=True, host='0.0.0.0', port=3000)