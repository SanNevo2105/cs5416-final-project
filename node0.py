"""
Start the server on node0

Node0 should handle requests from clients and distribute tasks to node1 and node2

Microservices on node0:
- Embedding service: Generate embeddings for queries
"""

import json
import time
from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
import requests

# import shared components from service.py
from service import TOTAL_NODES, NODE_NUMBER, NODE_0_IP, NODE_1_IP, NODE_2_IP, STEP_TO_NODEIP
from service import WORKERS, CONFIG, BATCH_SIZE, BATCH_WAIT_SECONDS
from service import FAISS_INDEX_PATH, DOCUMENTS_DIR
from service import PipelineRequest, PipelineResponse, PipelineData
from service import Embedding

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}             # request_id -> {"event": threading.Event(), "response": PipelineResponse, "count": int (how many requests are waiting for this result)}
results_lock = threading.Lock()

# multithreading on node0
# number of worker threads to run embedding service
THREADS = 3               # TRY DIFFERENT NUMBER OF THREADS!

# Node0 pipeline
pipeline = None

def worker():
    """
    worker thread function

    Continuously:
      - pull up to BATCH_SIZE requests from request_queue
      - do embedding
      - send to the node with step 2 service
    """
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

        reqs_data = [
            {"request_id": r["request_id"], "query": r["query"]}
            for r in batch
        ]

        # Create PipelineData objects
        reqs = [
            PipelineData(
                request_id=r['request_id'],
                query=r['query'],
                timestamp=time.time()
            )
            for r in reqs_data
        ]   

        # Process request
        responses = pipeline.process_batch(reqs)  

        # Prepare HTTP payload for step 2
        payload = {
            "requests": responses
        }   

        # send payload to step 2 node
        try:
            resp = requests.post(
                f"http://{STEP_TO_NODEIP[2]}/query",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            print(f"sent batch of size {len(batch)} to step 2")

        except Exception as e:
            for _ in batch:
                print(f"Error processing request: {e}")
       
        # Mark all batch items as done
        for _ in batch:
            request_queue.task_done()

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists and is finished
        with results_lock:
            # new request
            if request_id not in results:
                entry = {"event": threading.Event(), "response": None, "count": 1}
                results[request_id] = entry

                # Add to queue
                print(f"queueing request {request_id}")
                request_queue.put({
                    'request_id': request_id,
                    'query': query
                })
            
            # result already exists and is finished
            elif request_id in results and results[request_id]["event"].is_set():
                return jsonify(entry["response"]), 200
            
            # result is being processed
            elif request_id in results and not results[request_id]["event"].is_set():
                entry = results[request_id]
                entry["count"] += 1

        # wait for the result
        timeout = 300  # 5 minutes

        # time limit exceeded
        if not entry["event"].wait(timeout):
            with results_lock:
                results.pop(request_id, None)
            return jsonify({'error': 'Request timed out'}), 504
        
        # success, return the result
        with results_lock:
            entry["count"] -= 1
            if entry["count"] == 0:
                results.pop(request_id)
            return jsonify(entry["response"]), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/complete', methods=['POST'])
def complete():
    """Handle post request from step7 service"""
    data = request.json
    reqs_data = data.get('requests', [])            # a list of PipelineData

    # update results
    with results_lock:
        for res in reqs_data:
            results[res['request_id']]['response'] = PipelineResponse(
                request_id=res['request_id'],
                generated_response=res['generated_response'],
                sentiment=res['sentiment'],
                is_toxic=res['is_toxic'],
                processing_time=res['processing_time']
            )
            results[res['request_id']]['event'].set()

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
    Start the server on node0
    """
    assert NODE_NUMBER == 0, "This script should be run on node0 only."

    global pipeline

    print("="*60)
    print("NODE0 SERVER STARTING")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")

    print("Initializing pipeline...")
    pipeline = Embedding()
    print("Pipeline initialized!")

    # Start worker thread
    for i in range(THREADS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
    print(f"Started {THREADS} worker threads")

    hostport = NODE_0_IP

    hostname = hostport.split(':')[0]
    port = int(hostport.split(':')[1]) if ':' in hostport else 8000

    print(f"\nStarting Flask server on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()


