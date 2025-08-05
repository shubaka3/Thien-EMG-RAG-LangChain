# src/agent.py
import os
import logging
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from src.milvus_langchain import milvus_service
from src.database import get_db, log_to_db, AIModel

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None
    logging.warning("langchain_google_genai not installed. Gemini chat and embeddings will not be available.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_chat_llm(ai_info: AIModel, stream: bool = False):
    """Khởi tạo và trả về mô hình Chat LLM dựa trên AIModel."""
    if ai_info.provider == "openai":
        os.environ["OPENAI_API_KEY"] = ai_info.api_key
        return ChatOpenAI(
            temperature=0,
            model=ai_info.chat_model_name, # Dùng chat_model_name từ database
            api_key=ai_info.api_key,
            streaming=stream
        )
    elif ai_info.provider == "gemini":
        if not ChatGoogleGenerativeAI:
            raise ImportError("langchain_google_genai not installed. Cannot use Gemini chat.")
        if not ai_info.api_key: # Kiểm tra API key
            raise ValueError("Google API key is required for Gemini provider.")
        os.environ["GOOGLE_API_KEY"] = ai_info.api_key
        return ChatGoogleGenerativeAI(
            model=ai_info.chat_model_name, # Dùng chat_model_name từ database
            temperature=0,
            convert_system_message_to_human=True,
            api_key=ai_info.api_key,
            streaming=stream
        )
    else:
        raise ValueError(f"Unsupported chat LLM provider: {ai_info.provider}")


def search_document(query: str, milvus_collection_name: str, ai_info: AIModel) -> list[Document]:
    """
    Tìm kiếm tài liệu trong Milvus collection.
    Sử dụng thông tin mô hình embedding từ ai_info.
    """
    SEARCH_K_VALUE = int(os.getenv("SEARCH_K_VALUE", 4))

    logging.info(f"Searching for documents in collection '{milvus_collection_name}' with query: {query}")
    found_docs = milvus_service.search_documents(
        collection_name=milvus_collection_name,
        query=query,
        k=SEARCH_K_VALUE,
        embedding_model_provider=ai_info.provider,
        embedding_model_name=ai_info.embedding_model_name, # Dùng embedding_model_name từ ai_info
        api_key=ai_info.api_key,
        embedding_dim=ai_info.embedding_dim
    )
    return found_docs


def make_search_result_context(search_result: list[Document]) -> str:
    """Tạo ngữ cảnh từ kết quả tìm kiếm."""
    if not search_result:
        return "Không tìm thấy thông tin liên quan trong các tài liệu."
    
    context = ""
    for i, doc in enumerate(search_result):
        context += f"--- Nguồn {i+1} (trang {doc.metadata.get('page', 'N/A')}, file: {doc.metadata.get('source', 'N/A')}) ---\n"
        context += doc.page_content + "\n\n"
    return context


def invoke_agent(ai_info: AIModel, question: str, milvus_collection_name: str) -> dict:
    """
    Kích hoạt tác nhân AI để trả lời câu hỏi.
    Nhận đối tượng ai_info từ database.
    """
    db_session = next(get_db())
    try:
        llm = _get_chat_llm(ai_info, stream=False)

        prompt = """{context}

---

Dựa trên ngữ cảnh trên, hãy trả lời câu hỏi một cách tốt nhất có thể.

Câu hỏi: {question}

Trả lời: """

        search_result = search_document(question, milvus_collection_name, ai_info)
        result_context = make_search_result_context(search_result)

        chat_prompt = ChatPromptTemplate([('human', prompt)])
        chain = chat_prompt | llm

        response_content = chain.invoke({'context': result_context, 'question': question}).content

        # THAY ĐỔI: Sử dụng search_results thay vì sources
        log_to_db(db_session, ai_id=ai_info.id, question=question, answer=response_content, search_results=search_result)

        return {'answer': response_content, 'sources': search_result}
    except Exception as e:
        logging.error(f"Error invoking LLM for AI {ai_info.id}: {e}", exc_info=True)
        raise
    finally:
        db_session.close()


def stream_agent_response(ai_info: AIModel, question: str, milvus_collection_name: str):
    db_session = next(get_db())
    full_answer = ""
    search_result = []

    try:
        llm = _get_chat_llm(ai_info, stream=True)

        prompt = """{context}

---

Dựa trên ngữ cảnh trên, hãy trả lời câu hỏi một cách tốt nhất có thể.

Câu hỏi: {question}

Trả lời: """

        search_result = search_document(question, milvus_collection_name, ai_info)
        result_context = make_search_result_context(search_result)

        chat_prompt = ChatPromptTemplate([('human', prompt)])

        for chunk in llm.stream(chat_prompt.invoke({'context': result_context, 'question': question})):
            content = chunk.content or ""
            full_answer += content

            # CHỈNH SỬA ĐỂ STREAM ĐÚNG CHUẨN SSE
            yield f"data: {content}\n\n"

        # Kết thúc
        yield "data: [DONE]\n\n"

    except Exception as e:
        logging.error(f"Error streaming LLM response for AI {ai_info.id}: {e}", exc_info=True)
        yield f"data: ERROR: {str(e)}\n\n"
    finally:
        try:
            log_to_db(db_session, ai_id=ai_info.id, question=question, answer=full_answer, search_results=search_result)
        except Exception as log_e:
            logging.error(f"Error logging to DB after streaming: {log_e}", exc_info=True)
        db_session.close()
