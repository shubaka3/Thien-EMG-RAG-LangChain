chạy với lệnh : python -m src.main

cài .whl bị lỗi của pillow thì cài trực tiếp
pip install --only-binary=:all: Pillow==10.3.0

1 - khởi động server milvus docker và chạy service milvus

(venv) D:\Thien-EMG\Thien-rag-langchainV2>set PYTHONPATH=.
(venv) D:\Thien-EMG\Thien-rag-langchainV2>python src/milvus_api_service.py

sau đó truy cập 
http://127.0.0.1:5000/apidocs/ để test api
