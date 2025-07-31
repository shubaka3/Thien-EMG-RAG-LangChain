# main.py
import time
import os
import json
import uuid
from flask import Flask, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
load_dotenv('./.env')

from src.agent import invoke_agent, stream_agent_response
from src.load_file import list_sources_in_db, add_documents_to_db, load_documents,get_documents_from_db

app = Flask(__name__)

def make_response_chunk(chunk: str) -> str:
    """Create a response chunk in the format expected by OpenAI API."""
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
    data:dict = request.json
    
    if 'messages' not in data or not isinstance(data['messages'], list):
        return jsonify({'error': 'Message is required'}), 400
    question = data['messages'][0].get('content', '')
    if not question:
        return jsonify({'error': 'User question is required'}), 400

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
            'sources': [],
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
    uri = os.getenv('MILVUS_URL', 'http://localhost:19530')
    collection_name = os.getenv('MILVUS_COLLECTION', 'test_data2')
    
    sources = list_sources_in_db(uri, collection_name)
    
    return jsonify([{'source': source, 'split_count': sources[source]} for source in sources])

@app.route('/api/sources/update', methods=['POST'])
def update_source():
    add_documents_to_db(
        uri=os.getenv("MILVUS_URL", 'http://localhost:19530'),
        collection_name=os.getenv("MILVUS_COLLECTION", 'test_data2'),
        documents=load_documents(os.getenv("SERVER_STORAGE_PATH", './storage')),
        embedding_provider=os.getenv("SERVER_EMBEDDING_PROVIDER", 'openai'),
        model=os.getenv("SERVER_EMBEDDING_MODEL", 'text-embedding-3-large')
    )

    return jsonify({'status': 'success', 'message': 'Documents updated successfully.'})


@app.route('/api/sources/documents', methods=['GET'])
def fetch_documents():
    uri = os.getenv('MILVUS_URL', 'http://localhost:19530')
    collection_name = os.getenv('MILVUS_COLLECTION', 'test_data2')
    limit = int(request.args.get('limit', 20))

    try:
        docs = get_documents_from_db(uri, collection_name, limit)
        return jsonify([{'content': d.page_content, 'metadata': d.metadata} for d in docs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.debug = True
    port = os.getenv("SERVER_PORT", 3000)
    app.run(threaded=True, port=port, host='127.0.0.1')