# main.py
import time
import os
import json
import uuid
from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import requests # Thêm thư viện requests để gọi API
import logging
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from

load_dotenv('./.env')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.agent import invoke_agent, stream_agent_response # agent.py vẫn sử dụng MilvusService trực tiếp
from src.load_file import process_single_document, move_file, get_embedding_model # Import các hàm trợ giúp mới và get_embedding_model
from src.milvus_langchain import MilvusService # Để tương tác trực tiếp với Milvus

app = Flask(__name__)
swagger = Swagger(app) # Khởi tạo Swagger

# Cấu hình URL cho Milvus API Service
MILVUS_API_BASE_URL = os.getenv("MILVUS_API_BASE_URL", "http://127.0.0.1:5000/milvus")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY") # Lấy API Key cho Milvus API Service

# Cấu hình đường dẫn lưu trữ tệp
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", './uploads') # Để tải lên tệp tạm thời qua API
PENDING_FILES_PATH = os.getenv("PENDING_FILES_PATH", './pending_files') # Các tệp đang chờ được lập chỉ mục
PROCESSED_FILES_PATH = os.getenv("PROCESSED_FILES_PATH", './processed_storage') # Các tệp đã được lập chỉ mục thành công vào Milvus
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'test_data2') # Tên Milvus Collection mặc định

# Đảm bảo các thư mục tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PENDING_FILES_PATH, exist_ok=True)
os.makedirs(PROCESSED_FILES_PATH, exist_ok=True)

# Khởi tạo MilvusService cho các hoạt động trực tiếp (upload, ann_search)
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embeddings = get_embedding_model(
    provider=os.getenv("SERVER_EMBEDDING_PROVIDER", "openai"),
    model=EMBEDDING_MODEL
)
milvus_service = MilvusService(
    uri=os.getenv('MILVUS_URL', 'http://localhost:19530'),
    embedding_function=embeddings
)

# --- Hàm trợ giúp để thực hiện các yêu cầu xác thực tới Milvus API Service ---
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
            raise ValueError(f"Phương thức HTTP không được hỗ trợ: {method}")
        
        response.raise_for_status() # Ném ngoại lệ cho các mã trạng thái xấu (4xx hoặc 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API tại {url}: {e}")
        # Ném lại hoặc xử lý phù hợp
        raise

def make_response_chunk(chunk: str) -> str:
    """
    Tạo một chunk phản hồi theo định dạng mong đợi của OpenAI API.

    Args:
        chunk (str): Nội dung của chunk.

    Returns:
        str: Chuỗi JSON đã được serialize.
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
    Xử lý các yêu cầu trò chuyện, có thể là streaming hoặc non-streaming.
    """
    data:dict = request.json
    
    if 'messages' not in data or not isinstance(data['messages'], list):
        return jsonify({'error': 'Tin nhắn là bắt buộc'}), 400
    question = data['messages'][0].get('content', '')
    if not question:
        return jsonify({'error': 'Câu hỏi của người dùng là bắt buộc'}), 400

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
            'sources': [], # Có thể thêm nguồn ở đây nếu agent trả về chúng
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
    Truy xuất danh sách các nguồn tài liệu và số lượng của chúng từ Milvus API Service.
    Endpoint này vẫn sử dụng Milvus API Service.
    """
    try:
        response_data = _make_milvus_api_request('get', '/data/show_all')
        sources_data = response_data.get('source_counts', {})
        return jsonify([{'source': source, 'split_count': count} for source, count in sources_data.items()])
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API cho nguồn: {e}")
        return jsonify({'error': f'Không thể tìm nạp nguồn từ Milvus API: {e}'}), 500

@app.route('/api/upload_file', methods=['POST'])
@swag_from({
    'tags': ['Quản lý tệp'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Tệp để tải lên (PDF, Markdown, JSON).'
        }
    ],
    'responses': {
        200: {'description': 'Tệp đã tải lên thư mục pending để xử lý'},
        400: {'description': 'Không có phần tệp hoặc tệp không được phép'},
        500: {'description': 'Lỗi khi lưu tệp'}
    }
})
def upload_file():
    """
    Tải lên một tệp vào thư mục pending để xử lý sau này vào Milvus.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Không có phần tệp trong yêu cầu'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có tệp nào được chọn'}), 400

    if file:
        filename = secure_filename(file.filename)
        pending_file_path = os.path.join(PENDING_FILES_PATH, filename)
        processed_file_path = os.path.join(PROCESSED_FILES_PATH, filename)

        # Kiểm tra xem tệp có cùng tên đã tồn tại trong pending hay processed storage chưa
        if os.path.exists(pending_file_path):
            return jsonify({'message': f'Tệp {filename} đã tồn tại trong thư mục pending. Bỏ qua việc tải lên.'}), 200
        if os.path.exists(processed_file_path):
            return jsonify({'message': f'Tệp {filename} đã tồn tại trong thư mục processed storage. Bỏ qua việc tải lên.'}), 200

        try:
            file.save(pending_file_path)
            logging.info(f"Tệp '{filename}' đã được lưu vào '{PENDING_FILES_PATH}'.")
            return jsonify({'message': f'Tệp {filename} đã tải lên thư mục pending để xử lý.'}), 200
        except Exception as e:
            logging.error(f"Lỗi khi lưu tệp đã tải lên '{filename}': {e}")
            return jsonify({'error': f'Không thể lưu tệp: {e}'}), 500

@app.route('/api/sources/update', methods=['POST'])
@swag_from({
    'tags': ['Tài liệu Milvus'],
    'summary': 'Xử lý và lập chỉ mục các tài liệu mới từ thư mục pending vào Milvus.',
    'description': 'Quét thư mục pending để tìm các tài liệu mới, kiểm tra xem chúng đã tồn tại trong Milvus chưa, '
                   'lập chỉ mục các tài liệu mới và di chuyển các tệp đã xử lý vào thư mục storage. '
                   'Chỉ các tệp chưa có trong Milvus mới được chèn.',
    'responses': {
        200: {'description': 'Quá trình xử lý tài liệu đã bắt đầu thành công'},
        500: {'description': 'Lỗi trong quá trình xử lý tài liệu'}
    }
})
def update_source():
    """
    Xử lý các tài liệu từ thư mục pending, lập chỉ mục chúng vào Milvus,
    và di chuyển chúng đến thư mục processed storage.
    """
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    files_to_process = os.listdir(PENDING_FILES_PATH)
    
    if not files_to_process:
        return jsonify({'message': 'Không có tệp mới trong thư mục pending để xử lý.'}), 200

    logging.info(f"Bắt đầu xử lý {len(files_to_process)} tệp từ '{PENDING_FILES_PATH}'...")

    for filename in files_to_process:
        file_path = os.path.join(PENDING_FILES_PATH, filename)
        
        # Kiểm tra xem tệp đã được lập chỉ mục trong Milvus chưa (theo tên nguồn)
        # Chúng ta truy vấn MilvusService trực tiếp ở đây, không thông qua dịch vụ API
        if milvus_service.get_document_count_by_source(MILVUS_COLLECTION_NAME, filename) > 0:
            logging.info(f"Tệp '{filename}' đã tồn tại trong Milvus. Đang di chuyển đến processed storage.")
            move_file(file_path, os.path.join(PROCESSED_FILES_PATH, filename))
            skipped_count += 1
            continue

        try:
            # Xử lý tài liệu đơn lẻ để lấy các tài liệu Langchain với PK
            documents = process_single_document(file_path)
            if documents:
                # Thêm tài liệu vào Milvus trực tiếp thông qua MilvusService
                inserted_pks = milvus_service.add_documents(MILVUS_COLLECTION_NAME, documents)
                if inserted_pks:
                    logging.info(f"Đã chèn thành công {len(inserted_pks)} chunk cho tệp '{filename}'.")
                    # Di chuyển tệp đến processed storage sau khi chèn thành công
                    move_file(file_path, os.path.join(PROCESSED_FILES_PATH, filename))
                    processed_count += 1
                else:
                    logging.error(f"Không thể chèn tài liệu cho tệp '{filename}' vào Milvus.")
                    failed_count += 1
            else:
                logging.warning(f"Không có tài liệu nào được trích xuất từ tệp '{filename}'. Bỏ qua.")
                failed_count += 1 # Coi như thất bại nếu không trích xuất được tài liệu nào
                # Tùy chọn, di chuyển các tệp như vậy đến một thư mục 'failed'
                
        except Exception as e:
            logging.error(f"Lỗi khi xử lý tệp '{filename}': {e}", exc_info=True)
            failed_count += 1
            # Tùy chọn, di chuyển các tệp lỗi đến một thư mục 'failed' để kiểm tra thủ công

    return jsonify({
        'status': 'success',
        'message': f'Hoàn tất xử lý tài liệu.',
        'processed_files': processed_count,
        'skipped_files': skipped_count,
        'failed_files': failed_count
    }), 200

@app.route('/api/sources/delete_by_filename', methods=['POST'])
def delete_source_by_filename():
    """
    Xóa các tài liệu liên quan đến một tên tệp (nguồn) cụ thể thông qua Milvus API Service.
    """
    data: dict = request.json
    filename_to_delete = data.get('filename')

    if not filename_to_delete:
        return jsonify({'error': 'Tên tệp là bắt buộc để xóa tài liệu theo nguồn.'}), 400

    try:
        response_data = _make_milvus_api_request('post', '/documents/delete_by_source', json_data={'source': filename_to_delete})
        
        # Sau khi xóa thành công từ Milvus, xóa tệp khỏi processed_storage
        processed_file_path = os.path.join(PROCESSED_FILES_PATH, filename_to_delete)
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
            logging.info(f"Đã xóa tệp '{filename_to_delete}' khỏi processed_storage.")
        
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API để xóa theo tên tệp: {e}")
        return jsonify({'status': 'error', 'message': f'Không thể xóa tài liệu theo tên tệp thông qua Milvus API: {e}'}), 500

# --- Các Endpoint mới cho Quản lý Milvus Collection (qua Milvus API Service) ---
@app.route('/api/milvus/collections/list', methods=['GET'])
def list_milvus_collections_api():
    """
    Liệt kê tất cả các Milvus collection hiện có thông qua Milvus API Service.
    """
    try:
        response_data = _make_milvus_api_request('get', '/collections/list')
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API để liệt kê các collection: {e}")
        return jsonify({'error': f'Không thể liệt kê các Milvus collection: {e}'}), 500

@app.route('/api/milvus/collections/describe', methods=['GET'])
def describe_milvus_collection_api():
    """
    Mô tả một Milvus collection cụ thể thông qua Milvus API Service.
    Yêu cầu 'collection_name' làm tham số truy vấn.
    """
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Tên collection là bắt buộc.'}), 400
    try:
        response_data = _make_milvus_api_request('get', '/collections/describe', params={'collection_name': collection_name})
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API để mô tả collection '{collection_name}': {e}")
        return jsonify({'error': f"Không thể mô tả collection '{collection_name}': {e}"}), 500

@app.route('/api/milvus/collections/stats', methods=['GET'])
def get_milvus_collection_stats_api():
    """
    Lấy thống kê cho một Milvus collection cụ thể thông qua Milvus API Service.
    Yêu cầu 'collection_name' làm tham số truy vấn.
    """
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'Tên collection là bắt buộc.'}), 400
    try:
        response_data = _make_milvus_api_request('get', '/collections/stats', params={'collection_name': collection_name})
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gọi Milvus API để lấy thống kê cho collection '{collection_name}': {e}")
        return jsonify({'error': f"Không thể lấy thống kê cho collection '{collection_name}': {e}"}), 500


if __name__ == '__main__':
    # Đảm bảo collection mặc định được tạo khi khởi động ứng dụng chính
    # Đây là nơi tạo collection nếu nó chưa tồn tại, với auto_id=False cho PK
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 1536))
    try:
        milvus_service.create_collection(MILVUS_COLLECTION_NAME, VECTOR_DIMENSION, recreate=False)
        logging.info(f"Milvus collection '{MILVUS_COLLECTION_NAME}' đã được đảm bảo tồn tại.")
    except Exception as e:
        logging.error(f"Không thể đảm bảo Milvus collection '{MILVUS_COLLECTION_NAME}' tồn tại: {e}")
        # Rất quan trọng để thoát hoặc xử lý một cách duyên dáng nếu phụ thuộc dịch vụ cốt lõi thất bại khi khởi động
        exit(1)

    app.run(debug=True, host='0.0.0.0', port=5001) # Chạy ứng dụng Flask trên một cổng khác với Milvus API service
