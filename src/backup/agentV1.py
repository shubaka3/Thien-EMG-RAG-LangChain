# agent.py
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_openai import ChatOpenAI  # Model ngôn ngữ OpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def get_retreiver(uri:str, collection_name:str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name,
    )
    
    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )

def search_document(query: str) -> list[Document]:
    """Search for information in the provided document."""
    retriever = get_retreiver(os.getenv('MILVUS_URL', 'http://localhost:19530'), os.getenv('MILVUS_COLLECTION', 'test_data2'))
    results = retriever.invoke(query)
    
    if not results:
        return "No relevant information found."
    
    return results
    

def make_search_result_context(docs: list[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def invoke_agent(question: str) -> dict:
    # tool = create_retriever_tool(get_retreiver('http://localhost:19530', 'test_data2'), 'search_document', 'Search for information from provided document')
    llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=OPENAI_API_KEY)
    # question = "What does grade 6 science study about?"
    # question = question
    prompt = """{context}

    ---

    Given the context above, answer the question as best as possible.

    Question: {question}

    Answer: """
    print(f"Searching for information related to the question: {question}")
    search_result = search_document(question)
    result_context = ''
    if search_result:
        result_context = make_search_result_context(search_result)
    else:
        result_context = "No relevant information found."

    print(f"Sending search result to LLM")
    result = llm.invoke(ChatPromptTemplate([('human', prompt)]).invoke({'context': result_context, 'question': question}))
    return {
        'answer': result.content,
    }

def stream_agent_response(question: str):
    """Yields response tokens for streaming."""
    llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=OPENAI_API_KEY, stream=True)
    prompt = """{context}

---

Given the context above, answer the question as best as possible.

Question: {question}

Answer: """
    print(f"Searching for information related to the question: {question}")
    search_result = search_document(question)
    result_context = ''
    if search_result:
        result_context = make_search_result_context(search_result)
    else:
        result_context = "No relevant information found."

    print(f"Streaming search result to LLM")
    chat_prompt = ChatPromptTemplate([('human', prompt)])
    # Streaming response from LLM
    for chunk in llm.stream(chat_prompt.invoke({'context': result_context, 'question': question})):
        yield chunk.content
