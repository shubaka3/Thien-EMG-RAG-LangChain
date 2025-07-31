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

load_dotenv('./.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize embedding function and MilvusService once
# Ensure MILVUS_URL and MILVUS_COLLECTION are defined in .env
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
milvus_service = MilvusService(
    uri=os.getenv('MILVUS_URL', 'http://localhost:19530'),
    embedding_function=embeddings
)

# Get vector dimension from environment variable
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 1536))
SEARCH_K_VALUE = int(os.getenv('SEARCH_K_VALUE', 4))

def get_retriever(collection_name: str):
    """
    Creates a retriever from MilvusService.

    Args:
        collection_name (str): Name of the Milvus collection.

    Returns:
        object: A retriever object that can be used to search documents.
    """
    # Ensure collection exists before creating retriever
    # You can call create_collection here or elsewhere when the application starts
    # milvus_service.create_collection(collection_name, VECTOR_DIMENSION) # Create only if not exists
    
    # Langchain Milvus vectorstore has an as_retriever method
    vectorstore = milvus_service._get_vectorstore(collection_name)
    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": SEARCH_K_VALUE} # Use SEARCH_K_VALUE from .env
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
    results = milvus_service.search_documents(collection_name, query, k=SEARCH_K_VALUE) # Use SEARCH_K_VALUE
    
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
    llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'), api_key=OPENAI_API_KEY)
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
    return {
        'answer': result.content,
    }

def stream_agent_response(question: str):
    """
    Yields response tokens for streaming.

    Args:
        question (str): The user's question.

    Yields:
        str: Parts of the answer.
    """
    llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_COMPLETION_MODEL", 'gpt-4o-mini'), api_key=OPENAI_API_KEY, stream=True)
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

    print(f"Streaming search result to LLM")
    chat_prompt = ChatPromptTemplate([('human', prompt)])
    # Streaming response from LLM
    for chunk in llm.stream(chat_prompt.invoke({'context': result_context, 'question': question})):
        yield chunk.content

