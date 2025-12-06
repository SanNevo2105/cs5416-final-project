import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
import requests
import platform

# Detect macOS
if platform.system() == "Darwin":
    print("Detected macOS → setting FAISS to single-thread mode for stability")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
WORKERS = [NODE_0_IP, NODE_1_IP, NODE_2_IP]
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4)) 
BATCH_WAIT_SECONDS = os.environ.get("BATCH_WAIT_SECONDS", 0.05)

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512 # You must use this truncate length
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()

# Worker Counter
worker_counter = 0
worker_counter_lock = threading.Lock()

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float

class MonolithicPipeline:

    """
    Deliberately inefficient monolithic pipeline
    """
    
    def __init__(self):
        # adding metal and cuda support
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")
        # self.device = device
        self.device = torch.device("cpu")
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        print("Loading FAISS index into memory once for this process...")
        self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])
    
    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    def process_batch(self, requests: List[PipelineRequest]) -> List[PipelineResponse]:
        """
        Main pipeline execution for a batch of requests.
        """
        if not requests:
            return []

        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]

        print("\n" + "="*60)
        print(f"Processing batch of {batch_size} requests")
        print("="*60)
        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)

        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(
            queries,
            documents_batch
        )

        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(
            queries,
            reranked_docs_batch
        )

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)
        
        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} processed in {processing_time:.2f} seconds")
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineResponse(
                request_id=request.request_id,
                generated_response=responses_text[idx],
                sentiment=sentiments[idx],
                is_toxic=sensitivity_result,
                processing_time=processing_time
            ))
        
        return responses
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        model = SentenceTransformer(self.embedding_model_name).to(self.device)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        del model
        gc.collect()
        return embeddings
    
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        # print("Loading FAISS index")
        # index = faiss.read_index(CONFIG['faiss_index_path'])
        query_embeddings = query_embeddings.astype('float32')
        # _, indices = index.search(query_embeddings, CONFIG['retrieval_k'])
        # del index
        # gc.collect()
        _, indices = self.faiss_index.search(query_embeddings, CONFIG["retrieval_k"])
        return [row.tolist() for row in indices]
        

    
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        conn.close()
        return documents_batch
    
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        model.eval()
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        del model, tokenizer
        gc.collect()
        return reranked_batches
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        # model = AutoModelForCausalLM.from_pretrained(self.llm_model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        del model, tokenizer
        gc.collect()
        return responses
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        del classifier
        gc.collect()
        return sentiments
    
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        del classifier
        gc.collect()
        return toxicity_flags


# Global pipeline instance
pipeline = None

def worker_dispatcher(worker_index: int):
    """
    Continuously:
      - pull up to BATCH_SIZE requests from request_queue
      - send them as a batch to a worker via HTTP (round-robin)
      - store the results in `results` dict
    """
    global worker_counter
    while True:
        # Block until at least 1 request is available
        req = request_queue.get()
        if req is None:  # shutdown signal if you want one
            break

        batch = [req]

        # Try to grab up to BATCH_SIZE-1 more without blocking too long
        batch_deadline = time.time() + BATCH_WAIT_SECONDS
        while len(batch) < BATCH_SIZE:
            remaining = batch_deadline - time.time()
            if remaining <= 0:
                break
            try:
                # Wait a tiny bit for more requests to form a fuller batch
                more_req = request_queue.get(timeout=remaining)
                if more_req is None:
                    # push back the sentinel for other threads and stop
                    request_queue.put(None)
                    break
                batch.append(more_req)
            except Empty:
                break

        # Choose worker in round-robin, **per batch**
        # with worker_counter_lock:
        #     worker_index = worker_counter % len(WORKERS)
        #     worker_counter += 1
        worker_url = WORKERS[worker_index]
        print("Using worker", worker_index)

        payload = {
            "requests": [
                {"request_id": r["request_id"], "query": r["query"]}
                for r in batch
            ]
        }

        try:
            resp = requests.post(
                f"http://{worker_url}/process",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            responses = resp_json.get("responses", [])
            # Fan results back to waiting /query calls
            with results_lock:
                for res in responses:
                    results[res['request_id']] = {
                        'request_id': res['request_id'],
                        'generated_response': res['generated_response'],
                        'sentiment': res['sentiment'],
                        'is_toxic': res['is_toxic']
                    }
        except Exception as e:
            # On error, create error results for each request in the batch
            # responses = [
            #     {
            #         "request_id": r["request_id"],
            #         "error": f"Worker call failed: {e}",
            #     }
            #     for r in batch
            # ]
            for _ in batch:
                print(f"Error processing request: {e}")
       

        

        # Mark all batch items as done
        for _ in batch:
            request_queue.task_done()


# def process_requests_worker():
#     """Worker thread that processes requests from the queue"""
#     global pipeline
#     while True:
#         try:
#             request_data = request_queue.get()
#             if request_data is None:  # Shutdown signal
#                 break
            
#             # Create request object
#             req = PipelineRequest(
#                 request_id=request_data['request_id'],
#                 query=request_data['query'],
#                 timestamp=time.time()
#             )
            
#             # Process request
#             response = pipeline.process_request(req)
            
#             # Store result
#             with results_lock:
#                 results[request_data['request_id']] = {
#                     'request_id': response.request_id,
#                     'generated_response': response.generated_response,
#                     'sentiment': response.sentiment,
#                     'is_toxic': response.is_toxic
#                 }
            
#             request_queue.task_done()
#         except Exception as e:
#             print(f"Error processing request: {e}")
#             request_queue.task_done()



@app.route('/process', methods=['POST'])
def process_requests():
    """Worker thread that processes requests from the queue"""
    global pipeline
    
    data = request.json
    reqs_data = data.get('requests', [])

    # Create PipelineRequest objects
    reqs = [
        PipelineRequest(
            request_id=r['request_id'],
            query=r['query'],
            timestamp=time.time()
        )
        for r in reqs_data
    ]
    
    # Process request
    responses = pipeline.process_batch(reqs)

    # Convert PipelineResponse objects to dictionaries
    out = []
    for resp in responses:
        out.append({
            "request_id": resp.request_id,
            "generated_response": resp.generated_response,
            "sentiment": resp.sentiment,
            "is_toxic": resp.is_toxic,
        })

    return jsonify({"responses": out})
    


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    # global worker_counter
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200
        
        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({
            'request_id': request_id,
            'query': query
        })

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 300  # 5 minutes

        # worker_url = WORKERS[worker_counter%3]
        # worker_counter += 1

        # worker_resp = requests.post(worker_url + "/process", json=data, timeout=timeout )
        # worker_resp.raise_for_status()
        # return jsonify(worker_resp.json()), 200

        # busy waits until the request is processed
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200
            
            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            
            time.sleep(0.1)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    """
    Main execution function
    """
    global pipeline
    
    print("="*60)
    print("3 NODES MONOLITHIC CUSTOMER SUPPORT PIPELINE")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("\nNOTE: This is the basic implementation.")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()
    print("Pipeline initialized!")
    
    # Start worker thread
    # worker_thread = threading.Thread(target=worker_dispatcher, daemon=True) #idk what daemon is, hopefully not important
    # worker_thread.start()
    # print("Worker thread started!")
    if NODE_NUMBER == 0:
        for i in range(TOTAL_NODES):
            t = threading.Thread(target=worker_dispatcher, daemon=True, args=(i,))
            t.start()
        print(f"Started {TOTAL_NODES} dispatcher threads on router node")
    else:
        print("Worker node; not starting dispatcher thread")
    
    # Pick correct host:port for this node
    if NODE_NUMBER == 0:
        hostport = NODE_0_IP
    elif NODE_NUMBER == 1:
        hostport = NODE_1_IP
    else:
        hostport = NODE_2_IP

    hostname = hostport.split(':')[0]
    port = int(hostport.split(':')[1]) if ':' in hostport else 8000

    print(f"\nStarting Flask server on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()
