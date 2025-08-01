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

3:25PM - cập nhật agent,main
để dùng biến môi trường mỗi request thay vì khởi động lần đầu

OPENAI_API_KEY
OPENAI_COMPLETION_MODEL
SEARCH_K_VALUE
MILVUS_API_BASE_URL
MILVUS_API_KEY
CHUNK_SIZE 
CHUNK_OVERLAP
SEARCH_K_VALUE
TESSERACT_CMD_PATH


cách chạy sau khi pull code về

py -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

- 1 terminal python src/milvus_api_service.py trên win or
    python3 -m src.milvus_api_service unbutu

- 1 terminal python -m src.main
<<<<<<< HEAD
=======
9:31PM - Cập nhật luồng hoạt động của agent
https://docs.google.com/document/d/18zocGeH3WWpNeQrncDu62j3tS8svWEaOa7zPIlclBHw/edit?usp=sharing
>>>>>>> f7254f77a1cb3da8901ba4d3169d71290c5abbf3
