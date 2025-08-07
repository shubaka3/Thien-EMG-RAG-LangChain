# src/database.py
import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# Tải biến môi trường
load_dotenv('./.env')

# Cấu hình kết nối SQLAlchemy
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Định nghĩa Model bằng SQLAlchemy ---

class AIModel(Base):
    __tablename__ = "ai_models"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # user_id = Column(String, nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    provider = Column(String, nullable=False) # 'openai', 'gemini', 'custom'
    api_key = Column(String, nullable=False)
    
    tool = Column(String, nullable=True)  # NEW
    ai_domain = Column(String, nullable=True)                # NEW

    # THAY ĐỔI: Tách biệt tên mô hình chat và embedding
    embedding_model_name = Column(String, nullable=False) # Tên mô hình dùng cho embedding (ví dụ: 'text-embedding-ada-002')
    chat_model_name = Column(String, nullable=False)      # Tên mô hình dùng cho chat (ví dụ: 'gpt-3.5-turbo')
    
    embedding_dim = Column(Integer, nullable=False, default=1536)
    created_at = Column(DateTime, default=datetime.utcnow)

    collections = relationship("Collection", back_populates="ai", cascade="all, delete-orphan", passive_deletes=True)
    responses = relationship("AIResponse", back_populates="ai", cascade="all, delete-orphan", passive_deletes=True)
    user = relationship("User", back_populates="ai_models")



class Collection(Base):
    __tablename__ = "collections"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ai_id = Column(PG_UUID(as_uuid=True), ForeignKey("ai_models.id"), nullable=False)
    name = Column(String, nullable=False)
    milvus_collection_name = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ai = relationship("AIModel", back_populates="collections")

# --- Bảng ghi log ---

class AIResponse(Base):
    __tablename__ = 'ai_response'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ai_id = Column(PG_UUID(as_uuid=True), ForeignKey('ai_models.id'), nullable=False)
    question = Column(Text, nullable=False)
    response_metadata = Column(JSON)
    ai_answer = Column(Text)
    answertime = Column(DateTime, default=datetime.utcnow)
    create_at = Column(DateTime, default=datetime.utcnow)

    rag_progress = relationship("RagProgress", back_populates="response", cascade="all, delete-orphan", passive_deletes=True)
    ai = relationship("AIModel", back_populates="responses")

class RagProgress(Base):
    __tablename__ = 'rag_progress'
    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey('ai_response.id', ondelete='CASCADE'), nullable=False)
    ann_return = Column(Text)
    rag_metadata = Column(JSON)

    response = relationship("AIResponse", back_populates="rag_progress")

class User(Base):
    __tablename__ = "users"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    ai_models = relationship("AIModel", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)


# --- Hàm tiện ích ---

def get_db():
    """Dependency để lấy DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Tạo tất cả các bảng trong CSDL nếu chưa tồn tại."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

# --- Hàm ghi log đã được cập nhật để dùng SQLAlchemy ---
def log_to_db(db_session, ai_id: uuid.UUID, question: str, answer: str, search_results: list):
    """
    Ghi log câu hỏi và câu trả lời vào CSDL sử dụng session SQLAlchemy.
    """
    try:
        # Tạo bản ghi response chính
        new_response = AIResponse(
            ai_id=ai_id,
            question=question,
            ai_answer=answer,
            response_metadata={}
        )
        db_session.add(new_response)
        db_session.flush() # flush để lấy được new_response.id

        # Ghi các kết quả RAG
        for doc in search_results:
            rag_entry = RagProgress(
                question_id=new_response.id,
                ann_return=doc.page_content,
                rag_metadata=doc.metadata
            )
            db_session.add(rag_entry)
        
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"[DB LOG ERROR] {e}")