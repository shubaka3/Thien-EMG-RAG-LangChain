# main.py
import os
import uuid
import time
import json
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
import requests # Để dùng cho các đoạn code cũ hoặc gọi external API nếu có

# <<< THAY ĐỔI: Import các thành phần đã cập nhật
from src.database import SessionLocal, AIModel, Collection, create_tables, get_db
from src.agent import invoke_agent, stream_agent_response
from src.load_file import process_single_document, move_file # move_file có thể không cần nếu xóa file tạm
from src.milvus_langchain import milvus_service

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Khởi tạo Flask App và Swagger
app = Flask(__name__)
swagger = Swagger(app)

# Định nghĩa các thư mục upload/processed
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'temp_uploads')
PROCESSED_FILES_PATH = os.getenv('PROCESSED_FILES_PATH', 'processed_documents')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FILES_PATH, exist_ok=True)


# Để có thể sử dụng các endpoint này trong Swagger UI, bạn cần định nghĩa `definitions` cho Collection
app.config['SWAGGER'] = {
    'title': 'RAG API',
    'uiversion': 3,
    'definitions': {
        'AIModel': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string', 'format': 'uuid'},
                'user_id': {'type': 'string'},
                'name': {'type': 'string'},
                'provider': {'type': 'string'},
                'api_key': {'type': 'string'},
                'model_name': {'type': 'string'},
                'embedding_dim': {'type': 'integer'},
                'created_at': {'type': 'string', 'format': 'date-time'}
            }
        },
        'Collection': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string', 'format': 'uuid'},
                'ai_id': {'type': 'string', 'format': 'uuid'},
                'name': {'type': 'string'},
                'milvus_collection_name': {'type': 'string'},
                'created_at': {'type': 'string', 'format': 'date-time'}
            }
        },
        'AIResponse': {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'},
                'ai_id': {'type': 'string', 'format': 'uuid'},
                'question': {'type': 'string'},
                'response_metadata': {'type': 'object'},
                'ai_answer': {'type': 'string'},
                'answertime': {'type': 'string', 'format': 'date-time'},
                'create_at': {'type': 'string', 'format': 'date-time'}
            }
        },
        'RagProgress': {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'},
                'question_id': {'type': 'integer'},
                'ann_return': {'type': 'string'},
                'rag_metadata': {'type': 'object'}
            }
        }
    }
}


# Hàm tiện ích để tạo chunk phản hồi cho streaming
def make_response_chunk(content, model_name):
    if content is None: # Cuối stream
        return json.dumps({
            'choices': [{'delta': {}, 'finish_reason': 'stop'}],
            'model': model_name
        })
    else:
        return json.dumps({
            'choices': [{'delta': {'content': content}, 'finish_reason': None}],
            'model': model_name
        })


# --- API CHAT ĐÃ ĐƯỢC CẬP NHẬT VỚI BẢO MẬT ---

@app.route('/api/chat/completions', methods=['POST'])
def chat():
    """
    Xử lý chat, yêu cầu ai_id và user_id để xác thực.
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - messages
            - ai_id
            - user_id
            - collection_id
          properties:
            messages:
              type: array
              items:
                type: object
                properties:
                  role: {type: string}
                  content: {type: string}
            ai_id: {type: string, description: 'ID of the AI model.'}
            user_id: {type: string, description: 'ID of the user.'}
            collection_id: {type: string, description: 'ID of the collection to use.'}
            stream: {type: boolean, default: false, description: 'Set to true for streaming response.'}
    responses:
      200:
        description: Chat response
        schema:
          type: object
          properties:
            sources: {type: array, items: {type: string}}
            id: {type: string}
            object: {type: string}
            created: {type: integer}
            model: {type: string}
            choices:
              type: array
              items:
                type: object
                properties:
                  message:
                    type: object
                    properties:
                      role: {type: string}
                      content: {type: string}
                  finish_reason: {type: string}
            usage: {type: object}
      400: {description: 'Bad Request'}
      403: {description: 'Unauthorized or AI not found for this user'}
      404: {description: 'Collection not found for this AI'}
      500: {description: 'Internal Server Error'}
    tags:
      - Chat
    """
    data: dict = request.json
    
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    collection_id = data.get('collection_id')

    if not all([ai_id, user_id, collection_id]):
        return jsonify({'error': 'ai_id, user_id, and collection_id are required'}), 400
        
    question = data.get('messages', [{}])[-1].get('content')
    if not question:
        return jsonify({'error': 'User question is required'}), 400

    db = SessionLocal()
    try:
        # 1. Xác thực AI thuộc về User
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or AI not found for this user'}), 403

        # 2. Xác thực Collection thuộc về AI
        collection_info = db.query(Collection).filter(Collection.id == collection_id, Collection.ai_id == ai_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found for this AI'}), 404
        
        # Lấy các thông tin cần thiết
        api_key = ai_info.api_key
        model_name = ai_info.model_name
        provider = ai_info.provider
        milvus_collection = collection_info.milvus_collection_name

    except Exception as e:
        logging.error(f"Database error while fetching AI config: {e}")
        return jsonify({'error': 'Could not retrieve AI configuration'}), 500
    finally:
        db.close()

    if data.get('stream', False):
        @stream_with_context
        def generate():
            for chunk in stream_agent_response(ai_info.id, question, api_key, model_name, provider, milvus_collection):
                yield f"data: {json.dumps(chunk)}\n\n" # Đảm bảo chunk đã là dict/json serializable
            yield "data: [DONE]\n\n"
        return Response(generate(), mimetype='text/event-stream')
    else:
        answer = invoke_agent(ai_info.id, question, api_key, model_name, provider, milvus_collection)
        return jsonify({
            'sources': answer.get('sources', []),
            'id': f'emgcmpl-{uuid.uuid4()}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model_name,
            'choices': [{'message': {'role': 'assistant', 'content': answer['answer']}, 'finish_reason': 'stop'}],
            'usage': {
                'prompt_tokens': len(question.split()),
                'completion_tokens': len(answer['answer'].split()),
                'total_tokens': len(question.split()) + len(answer['answer'].split())
            },
        })

# --- API QUẢN LÝ AI MODELS ---

@app.route('/api/ai', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'description': 'ID của người dùng sở hữu AI.'},
                    'name': {'type': 'string', 'description': 'Tên của AI model.'},
                    'provider': {'type': 'string', 'enum': ['openai', 'gemini', 'custom'], 'description': 'Nhà cung cấp AI.'},
                    'api_key': {'type': 'string', 'description': 'API Key của nhà cung cấp.'},
                    'model_name': {'type': 'string', 'description': 'Tên model cụ thể (ví dụ: gpt-3.5-turbo).'},
                    'embedding_dim': {'type': 'integer', 'default': 1536, 'description': 'Kích thước embedding dimension.'}
                },
                'required': ['user_id', 'name', 'provider', 'api_key', 'model_name']
            }
        }
    ],
    'responses': {
        201: {'description': 'AI model đã được tạo thành công.', 'schema': {'type': 'object', 'properties': {'id': {'type': 'string'}}}},
        400: {'description': 'Bad Request'},
        500: {'description': 'Internal Server Error'}
    }
})
def create_ai():
    """
    Tạo một AI model mới.
    ---
    tags:
      - AI Management
    """
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    provider = data.get('provider')
    api_key = data.get('api_key')
    model_name = data.get('model_name')
    embedding_dim = data.get('embedding_dim', 1536)

    if not all([user_id, name, provider, api_key, model_name]):
        return jsonify({'error': 'Missing required fields'}), 400

    db = next(get_db())
    try:
        new_ai = AIModel(
            user_id=user_id,
            name=name,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            embedding_dim=embedding_dim
        )
        db.add(new_ai)
        db.commit()
        db.refresh(new_ai)
        return jsonify({'message': 'AI model created successfully', 'id': str(new_ai.id)}), 201
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating AI model: {e}", exc_info=True)
        return jsonify({'error': f'Failed to create AI model: {e}'}), 500
    finally:
        db.close()


# --- API QUẢN LÝ COLLECTIONS ---

@app.route('/api/collections', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'ai_id': {'type': 'string', 'description': 'ID của AI model mà collection này thuộc về.'},
                    'user_id': {'type': 'string', 'description': 'ID của người dùng sở hữu AI (để xác thực).'},
                    'name': {'type': 'string', 'description': 'Tên của collection (ví dụ: "Tài liệu kỹ thuật", "Hợp đồng").'},
                    'milvus_collection_name': {'type': 'string', 'description': 'Tên collection duy nhất trong Milvus.'}
                },
                'required': ['ai_id', 'user_id', 'name', 'milvus_collection_name']
            }
        }
    ],
    'responses': {
        201: {'description': 'Collection đã được tạo thành công.', 'schema': {'type': 'object', 'properties': {'id': {'type': 'string'}}}},
        400: {'description': 'Bad Request'},
        403: {'description': 'Unauthorized or AI not found for this user'},
        409: {'description': 'Milvus collection name already exists'},
        500: {'description': 'Internal Server Error'}
    }
})
def create_collection():
    """
    Tạo một collection mới liên kết với một AI Model.
    ---
    tags:
      - Collection Management
    """
    data = request.json
    ai_id = data.get('ai_id')
    user_id = data.get('user_id')
    name = data.get('name')
    milvus_collection_name = data.get('milvus_collection_name')

    if not all([ai_id, user_id, name, milvus_collection_name]):
        return jsonify({'error': 'Missing required fields'}), 400

    db = next(get_db())
    try:
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or AI not found for this user'}), 403

        existing_collection = db.query(Collection).filter(Collection.milvus_collection_name == milvus_collection_name).first()
        if existing_collection:
            return jsonify({'error': f'Milvus collection name "{milvus_collection_name}" already exists for another collection.'}), 409

        logging.info(f"Attempting to create Milvus collection '{milvus_collection_name}' with dim {ai_info.embedding_dim}")
        milvus_service.create_collection(milvus_collection_name, ai_info.embedding_dim)
        logging.info(f"Milvus collection '{milvus_collection_name}' created or already exists.")

        new_collection = Collection(
            ai_id=ai_id,
            name=name,
            milvus_collection_name=milvus_collection_name
        )
        db.add(new_collection)
        db.commit()
        db.refresh(new_collection)
        return jsonify({'message': 'Collection created successfully', 'id': str(new_collection.id)}), 201
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating collection: {e}", exc_info=True)
        return jsonify({'error': f'Failed to create collection: {e}'}), 500
    finally:
        db.close()


@app.route('/api/collections/<ai_id>', methods=['GET'])
@swag_from({
    'parameters': [
        {
            'name': 'ai_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'ID của AI model.'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'ID của người dùng (để xác thực).'
        }
    ],
    'responses': {
        200: {'description': 'Danh sách các collections.', 'schema': {'type': 'array', 'items': {'$ref': '#/definitions/Collection'}}},
        400: {'description': 'Bad Request'},
        403: {'description': 'Unauthorized or AI not found for this user'}
    }
})
def get_collections_for_ai(ai_id):
    """
    Lấy danh sách các collections thuộc về một AI model.
    ---
    tags:
      - Collection Management
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        ai_info = db.query(AIModel).filter(AIModel.id == ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or AI not found for this user'}), 403

        collections = db.query(Collection).filter(Collection.ai_id == ai_id).all()
        return jsonify([
            {
                'id': str(c.id),
                'ai_id': str(c.ai_id),
                'name': c.name,
                'milvus_collection_name': c.milvus_collection_name,
                'created_at': c.created_at.isoformat()
            } for c in collections
        ]), 200
    except Exception as e:
        logging.error(f"Error fetching collections for AI {ai_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to fetch collections: {e}'}), 500
    finally:
        db.close()


@app.route('/api/collections/detail/<collection_id>', methods=['GET'])
@swag_from({
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'ID của Collection.'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'ID của người dùng (để xác thực).'
        }
    ],
    'responses': {
        200: {'description': 'Thông tin chi tiết collection.', 'schema': {'$ref': '#/definitions/Collection'}},
        400: {'description': 'Bad Request'},
        403: {'description': 'Unauthorized or Collection not found for this user'},
        404: {'description': 'Collection not found'}
    }
})
def get_collection_detail(collection_id):
    """
    Lấy thông tin chi tiết của một collection.
    ---
    tags:
      - Collection Management
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or Collection not found for this user'}), 403

        return jsonify({
            'id': str(collection_info.id),
            'ai_id': str(collection_info.ai_id),
            'name': collection_info.name,
            'milvus_collection_name': collection_info.milvus_collection_name,
            'created_at': collection_info.created_at.isoformat()
        }), 200
    except Exception as e:
        logging.error(f"Error fetching collection detail for {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to fetch collection detail: {e}'}), 500
    finally:
        db.close()


@app.route('/api/collections/<collection_id>', methods=['DELETE'])
@swag_from({
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'ID của Collection cần xóa.'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'ID của người dùng (để xác thực quyền).'
        }
    ],
    'responses': {
        200: {'description': 'Collection và dữ liệu Milvus liên quan đã được xóa thành công.'},
        400: {'description': 'Bad Request'},
        403: {'description': 'Unauthorized or Collection not found for this user'},
        404: {'description': 'Collection not found'},
        500: {'description': 'Internal Server Error'}
    }
})
def delete_collection(collection_id):
    """
    Xóa một collection và tất cả dữ liệu liên quan trong Milvus.
    ---
    tags:
      - Collection Management
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        collection_to_delete = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_to_delete:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_to_delete.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or Collection not found for this user'}), 403

        milvus_collection_name = collection_to_delete.milvus_collection_name

        logging.info(f"Attempting to drop Milvus collection '{milvus_collection_name}'")
        milvus_service.drop_collection(milvus_collection_name)
        logging.info(f"Milvus collection '{milvus_collection_name}' dropped.")

        db.delete(collection_to_delete)
        db.commit()

        return jsonify({'message': f'Collection {collection_id} and its associated Milvus data deleted successfully.'}), 200
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to delete collection: {e}'}), 500
    finally:
        db.close()

# --- API TẢI LÊN VÀ XỬ LÝ TÀI LIỆU CHO COLLECTION CỤ THỂ ---

@app.route('/api/documents/upload', methods=['POST'])
@swag_from({
    'tags': ['Document Management'],
    'summary': 'Upload and process a document for a specific collection.',
    'description': 'Uploads a file, processes it (chunking, embedding), and adds to the specified Milvus collection.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'The document file to upload (e.g., PDF, TXT, DOCX).'
        },
        {
            'name': 'collection_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'The ID of the collection to add the document to.'
        },
        {
            'name': 'user_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': 'The ID of the user (for authentication and authorization).'
        }
    ],
    'responses': {
        200: {'description': 'Document uploaded and processing initiated.'},
        400: {'description': 'Bad Request (missing file, collection_id, or user_id).'},
        403: {'description': 'Unauthorized (user does not own the AI/Collection).'},
        404: {'description': 'Collection not found.'},
        409: {'description': 'File with the same name already exists in the collection.'},
        500: {'description': 'Internal Server Error.'}
    }
})
def upload_document_to_collection():
    """
    Uploads a document and processes it into a specific Milvus collection.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    collection_id = request.form.get('collection_id')
    user_id = request.form.get('user_id')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not all([collection_id, user_id]):
        return jsonify({'error': 'collection_id and user_id are required in form data'}), 400

    db = next(get_db())
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or AI not found for this user (and its collection)'}), 403

        milvus_collection_name = collection_info.milvus_collection_name

        if milvus_service.get_document_count_by_source(milvus_collection_name, filename) > 0:
            return jsonify({'message': f'File "{filename}" already exists in collection "{collection_info.name}". Skipping upload.'}), 409

        file.save(temp_file_path)
        logging.info(f"File '{filename}' saved temporarily to '{UPLOAD_FOLDER}'.")

        logging.info(f"Processing file '{filename}' for collection '{milvus_collection_name}'...")
        
        documents = process_single_document(temp_file_path)
        if documents:
            for doc in documents:
                doc.metadata['source'] = filename # Ensure source is set
                # Nếu có metadata tùy chỉnh từ request hoặc logic khác, thêm vào đây
                # doc.metadata['custom_field'] = 'value' 

            inserted_pks = milvus_service.add_documents(
                milvus_collection_name,
                documents,
                embedding_model_provider=ai_info.provider,
                embedding_model_name=ai_info.model_name,
                api_key=ai_info.api_key,
                embedding_dim=ai_info.embedding_dim
            )
            if inserted_pks:
                logging.info(f"Successfully inserted {len(inserted_pks)} chunks for file '{filename}' into Milvus collection '{milvus_collection_name}'.")
                # Move file to processed storage after successful insertion
                move_file(temp_file_path, os.path.join(PROCESSED_FILES_PATH, filename))
                return jsonify({
                    'message': f'File "{filename}" processed and added to collection "{collection_info.name}" successfully.',
                    'inserted_chunks': len(inserted_pks)
                }), 200
            else:
                logging.error(f"Failed to insert documents for file '{filename}' into Milvus.")
                os.remove(temp_file_path)
                return jsonify({'error': f'Failed to process and insert document "{filename}".'}), 500
        else:
            logging.warning(f"No documents extracted from file '{filename}'. Skipping.")
            os.remove(temp_file_path)
            return jsonify({'message': f'No extractable content found in file "{filename}".'}), 200

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logging.error(f"Error processing document '{filename}': {e}", exc_info=True)
        return jsonify({'error': f'Failed to upload and process document: {e}'}), 500
    finally:
        db.close()


# --- API MỚI CHO QUẢN LÝ TÀI LIỆU TRONG COLLECTION ---

@app.route('/api/collections/<collection_id>/documents', methods=['GET'])
@swag_from({
    'tags': ['Document Management'],
    'summary': 'List all documents (filenames) in a specific collection.',
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The ID of the collection.'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'The ID of the user (for authentication and authorization).'
        }
    ],
    'responses': {
        200: {'description': 'List of document filenames.', 'schema': {'type': 'array', 'items': {'type': 'string'}}},
        400: {'description': 'Bad Request.'},
        403: {'description': 'Unauthorized.'},
        404: {'description': 'Collection not found.'},
        500: {'description': 'Internal Server Error.'}
    }
})
def list_documents_in_collection(collection_id):
    """
    Lists all unique filenames (sources) that have been indexed into a specific Milvus collection.
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or Collection not found for this user'}), 403

        milvus_collection_name = collection_info.milvus_collection_name
        
        filenames = milvus_service.get_all_sources_in_collection(milvus_collection_name)
        
        return jsonify(filenames), 200
    except Exception as e:
        logging.error(f"Error listing documents for collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to list documents: {e}'}), 500
    finally:
        db.close()

@app.route('/api/collections/<collection_id>/documents/<path:filename>/chunks', methods=['GET'])
@swag_from({
    'tags': ['Document Management'],
    'summary': 'Retrieve all chunks for a specific document in a collection.',
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The ID of the collection.'
        },
        {
            'name': 'filename',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The filename of the document (full name, e.g., "my_doc.pdf").'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'The ID of the user (for authentication and authorization).'
        }
    ],
    'responses': {
        200: {'description': 'List of document chunks with content and metadata.', 'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'text': {'type': 'string'}, 'source': {'type': 'string'}, 'metadata': {'type': 'object'}}}}},
        400: {'description': 'Bad Request.'},
        403: {'description': 'Unauthorized.'},
        404: {'description': 'Collection or document not found.'},
        500: {'description': 'Internal Server Error.'}
    }
})
def get_document_chunks(collection_id, filename):
    """
    Retrieves all text chunks and their metadata for a specific document within a collection.
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or Collection not found for this user'}), 403

        milvus_collection_name = collection_info.milvus_collection_name

        chunks = milvus_service.get_chunks_by_source(milvus_collection_name, filename)
        
        if not chunks:
            return jsonify({'message': f'No chunks found for document "{filename}" in collection "{collection_info.name}".'}), 404
        
        return jsonify(chunks), 200
    except Exception as e:
        logging.error(f"Error getting chunks for document '{filename}' in collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to retrieve document chunks: {e}'}), 500
    finally:
        db.close()


@app.route('/api/collections/<collection_id>/documents/<path:filename>', methods=['DELETE'])
@swag_from({
    'tags': ['Document Management'],
    'summary': 'Delete a specific document and its chunks from a collection.',
    'parameters': [
        {
            'name': 'collection_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The ID of the collection.'
        },
        {
            'name': 'filename',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The filename of the document to delete (full name, e.g., "my_doc.pdf").'
        },
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'The ID of the user (for authentication and authorization).'
        }
    ],
    'responses': {
        200: {'description': 'Document and its chunks deleted successfully.'},
        400: {'description': 'Bad Request.'},
        403: {'description': 'Unauthorized.'},
        404: {'description': 'Collection or document not found.'},
        500: {'description': 'Internal Server Error.'}
    }
})
def delete_document_from_collection(collection_id, filename):
    """
    Deletes a specific document (all its chunks from Milvus) and its physical file from storage.
    """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as a query parameter'}), 400

    db = next(get_db())
    try:
        collection_info = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection_info:
            return jsonify({'error': 'Collection not found'}), 404

        ai_info = db.query(AIModel).filter(AIModel.id == collection_info.ai_id, AIModel.user_id == user_id).first()
        if not ai_info:
            return jsonify({'error': 'Unauthorized or Collection not found for this user'}), 403

        milvus_collection_name = collection_info.milvus_collection_name

        deleted_count = milvus_service.delete_documents_by_source(milvus_collection_name, filename)
        
        if deleted_count > 0:
            # Delete physical file from processed storage
            processed_file_path = os.path.join(PROCESSED_FILES_PATH, filename)
            if os.path.exists(processed_file_path):
                os.remove(processed_file_path)
                logging.info(f"Deleted physical file '{filename}' from processed storage.")
            
            return jsonify({'message': f'Document "{filename}" and its {deleted_count} chunks deleted successfully from collection "{collection_info.name}".'}), 200
        else:
            return jsonify({'message': f'Document "{filename}" not found in collection "{collection_info.name}" or no chunks to delete.'}), 404
            
    except Exception as e:
        logging.error(f"Error deleting document '{filename}' from collection {collection_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to delete document: {e}'}), 500
    finally:
        db.close()


if __name__ == '__main__':
    create_tables()
    app.run(debug=True, port=3000)