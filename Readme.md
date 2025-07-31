chạy với lệnh : python -m src.main

cài .whl bị lỗi của pillow thì cài trực tiếp
pip install --only-binary=:all: Pillow==10.3.0

1 - khởi động server milvus docker và chạy service milvus

(venv) D:\Thien-EMG\Thien-rag-langchainV2>set PYTHONPATH=.
(venv) D:\Thien-EMG\Thien-rag-langchainV2>python src/milvus_api_service.py

sau đó truy cập 
http://127.0.0.1:5000/apidocs/ để test api


7:57:AM - 31/7 
- milvus_langchain.py (Cải tiến quản lý Collection) - caching cho collection, tránh tải, giải phóng liên tục

2:46:PM - 31/7
- Sửa lại hầu hết rất nhiều và đã test chạy tốt
- test endpoint:  https://docs.google.com/spreadsheets/d/1ATC2GXBlrS709IgowpO4G2Ak4OyHodTBtWZvOWQs2QQ/edit?usp=sharing
đa số là chạy tốt và sẽ cải thiện thêm 