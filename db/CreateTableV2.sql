-- file: create_all_tables.sql

-- Tạo bảng ai_models
CREATE TABLE ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    provider VARCHAR(255) NOT NULL, -- 'openai', 'gemini', 'custom'
    api_key VARCHAR(255) NOT NULL,
    embedding_model_name VARCHAR(255) NOT NULL, -- Đổi tên từ model_name
    chat_model_name VARCHAR(255) NOT NULL,      -- Cột mới thêm
    embedding_dim INTEGER NOT NULL DEFAULT 1536,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tạo chỉ mục cho user_id trên ai_models
CREATE INDEX ix_ai_models_user_id ON ai_models (user_id);

-- Tạo bảng collections
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    milvus_collection_name VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_collections_ai_id FOREIGN KEY (ai_id) REFERENCES ai_models(id) ON DELETE CASCADE
);

-- Tạo chỉ mục cho milvus_collection_name trên collections
CREATE INDEX ix_collections_milvus_collection_name ON collections (milvus_collection_name);

-- Tạo bảng ai_response
CREATE TABLE ai_response (
    id SERIAL PRIMARY KEY,
    ai_id UUID NOT NULL,
    question TEXT NOT NULL,
    response_metadata JSONB, -- Đổi từ 'metadata' thành 'response_metadata' để tránh nhầm lẫn
    ai_answer TEXT,
    answertime TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    create_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ai_response_ai_id FOREIGN KEY (ai_id) REFERENCES ai_models(id) ON DELETE CASCADE
);

-- Tạo bảng rag_progress
CREATE TABLE rag_progress (
    id SERIAL PRIMARY KEY,
    question_id INTEGER NOT NULL,
    ann_return TEXT,
    rag_metadata JSONB, -- Đổi từ 'metadataa' thành 'rag_metadata' để rõ ràng hơn
    CONSTRAINT fk_rag_progress_question_id FOREIGN KEY (question_id) REFERENCES ai_response(id) ON DELETE CASCADE
);