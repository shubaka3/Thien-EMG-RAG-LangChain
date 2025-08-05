# agent.py
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from src.milvus_langchain import MilvusService # Import the new MilvusService
from src.db_logger import log_to_db
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv('./.env')

# Initialize embedding function and MilvusService once (these still require restart if their config changes)
# EMBEDDING_MODEL is read here, if it changes, the 'embeddings' object needs to be re-created, requiring a restart.
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# MILVUS_URL is read here for MilvusService initialization, if it changes, 'milvus_service' needs to be re-created, requiring a restart.
milvus_service = MilvusService(
    uri=os.getenv('MILVUS_URL', 'http://localhost:19530'),
    embedding_function=embeddings
)

# VECTOR_DIMENSION is used for collection creation, which is a startup task
# Its change requires a restart if the collection needs to be recreated.
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 1536))

def get_retriever(collection_name: str):
    """
    Creates a retriever from MilvusService.

    Args:
        collection_name (str): Name of the Milvus collection.

    Returns:
        object: A retriever object that can be used to search documents.
    """
    # Read SEARCH_K_VALUE dynamically here
    SEARCH_K_VALUE = int(os.getenv('SEARCH_K_VALUE', 4))
    
    # Langchain Milvus vectorstore has an as_retriever method
    vectorstore = milvus_service._get_vectorstore(collection_name)
    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": SEARCH_K_VALUE}
    )

def search_document(query: str) -> list[Document]:
    """
    Searches for information in the provided document via Milvus.

    Args:
        query (str): The query string to search for.

    Returns:
        list[Document]: List of matching Langchain documents.
    """
    collection_name = os.getenv('MILVUS_COLLECTION', 'test_data2')
    # Read SEARCH_K_VALUE dynamically here
    SEARCH_K_VALUE = int(os.getenv('SEARCH_K_VALUE', 4))
    results = milvus_service.search_documents(collection_name, query, k=SEARCH_K_VALUE)
    
    if not results:
        return [] # Return empty list instead of string
    
    return results
    

def make_search_result_context(docs: list[Document]) -> str:
    """
    Creates a context string from a list of documents.

    Args:
        docs (list[Document]): List of Langchain Document objects.

    Returns:
        str: Context string concatenated from the content of the documents.
    """
    return "\n\n".join([doc.page_content for doc in docs])


def invoke_agent(question: str) -> dict:
    """
    Invokes the agent to answer a question based on search context.

    Args:
        question (str): The user's question.

    Returns:
        dict: A dictionary containing the answer.
    """
    # Read OPENAI_API_KEY and OPENAI_COMPLETION_MODEL dynamically here
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini')

    llm = ChatOpenAI(temperature=0, model=OPENAI_COMPLETION_MODEL, api_key=OPENAI_API_KEY)
    prompt = """{context}

    ---

    Dựa trên ngữ cảnh trên, hãy trả lời câu hỏi một cách tốt nhất có thể.

    Câu hỏi: {question}

    Trả lời: """
    print(f"Searching for information related to the question: {question}")
    search_result = search_document(question)
    result_context = ''
    if search_result:
        result_context = make_search_result_context(search_result)
    else:
        result_context = "Không tìm thấy thông tin liên quan."

    print(f"Sending search result to LLM")
    result = llm.invoke(ChatPromptTemplate([('human', prompt)]).invoke({'context': result_context, 'question': question}))
    
    answer_text = result.content
    print("📝 Ghi log vào cơ sở dữ liệu...")
    logging.info("📝 Ghi log vào cơ sở dữ liệu...")
    try:
        log_to_db(question, answer_text, search_result)
    except Exception as e:
        print(f"[⚠️ DB ERROR] Không thể ghi log: {e}")
        logging.info(f"⚠️ DB ERROR] Không thể ghi log: {e}")

    return {
        'answer': result.content,
    }

def stream_agent_response(question: str):
    """
    Yields response tokens for streaming.
    After streaming completes, logs the full answer to database.

    Args:
        question (str): The user's question.

    Yields:
        str: Parts of the answer.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini')

    llm = ChatOpenAI(temperature=0, model=OPENAI_COMPLETION_MODEL, api_key=OPENAI_API_KEY, stream=True)
    prompt = """{context}

---

Dựa trên ngữ cảnh trên, hãy trả lời câu hỏi một cách tốt nhất có thể.

Câu hỏi: {question}

Trả lời: """

    print(f"Searching for information related to the question: {question}")
    search_result = search_document(question)
    result_context = make_search_result_context(search_result) if search_result else "Không tìm thấy thông tin liên quan."

    print("Streaming search result to LLM")
    chat_prompt = ChatPromptTemplate([('human', prompt)])
    
    # Tích lũy kết quả để log sau
    full_answer = ""

    try:
        for chunk in llm.stream(chat_prompt.invoke({'context': result_context, 'question': question})):
            content = chunk.content or ""
            full_answer += content
            yield content  # Stream từng phần ra ngoài
    finally:
        # Sau khi stream xong thì log lại
        print("📝 Ghi log vào cơ sở dữ liệu sau khi stream xong...")
        logging.info("📝 Ghi log vào cơ sở dữ liệu sau khi stream xong...")
        try:
            log_to_db(question, full_answer, search_result)
            logging.info(f"Data log {question} - {full_answer[:50]} - {search_result}... đã được ghi vào cơ sở dữ liệu.")

        except Exception as e:
            print(f"[⚠️ DB ERROR] Không thể ghi log sau stream: {e}")
            logging.warning(f"[⚠️ DB ERROR] Không thể ghi log sau stream: {e}")
