"""
Start the server on node2

Microservices on node2:
- LLM answer generation
"""
import json
import multiprocessing
import time
from flask import Flask, request, jsonify
from queue import Empty
import requests
from dataclasses import asdict

# import shared components from service.py
from service import TOTAL_NODES, NODE_NUMBER, NODE_0_IP, NODE_1_IP, NODE_2_IP, STEP_TO_NODEIP
from service import WORKERS, CONFIG, BATCH_SIZE, BATCH_WAIT_SECONDS
from service import FAISS_INDEX_PATH, DOCUMENTS_DIR
from service import PipelineRequest, PipelineResponse, PipelineData
from service import ResponseGeneration

# Flask app
app = Flask(__name__)

# Request queue 
request_queue = multiprocessing.Queue()

# multiprocessing on node2
# number of worker processes to run node2 services
PROCESSES = 2               # TRY DIFFERENT NUMBER OF PROCESSES!

def worker(request_queue):
    """
    worker function

    Continuously:
      - pull up to BATCH_SIZE requests from request_queue
      - do node 2 services
      - send result to node 0 to complete the pipeline
    """
    pipeline = ResponseGeneration()
    # print("Node2: initialized ResponseGeneration pipeline")

    while True:
        # Block until at least 1 request is available
        req = request_queue.get()
        # print("Node2: got request from queue")
        if req is None:  # shutdown signal if you want one
            break

        

        # a list of PipelineData
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
                    # push back the sentinel for other processes and stop
                    request_queue.put(None)
                    break
                batch.append(more_req)
            except Empty:
                break

        # Process request
        responses = pipeline.process_batch(batch)  
        responses = [asdict(r) for r in responses]

        # Prepare HTTP payload for next step
        payload = {
            "requests": responses
        }   

        # send payload to node 0
        try:
            resp = requests.post(
                f"http://{NODE_0_IP}/complete",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            print(f"sent batch of size {len(batch)} to node 0")

        except Exception as e:
            for _ in batch:
                print(f"Error processing request: {e}")


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    global request_queue
    try:
        data = request.json
        pipelinedata_requests = data.get('requests')     # a list of PipelineData from previous step
        pipelinedata_requests = [
            PipelineData(**r) 
            for r in pipelinedata_requests
        ]  # convert dict to PipelineData

        # put requests to queue
        for pipelinedata in pipelinedata_requests:
            # print("Node2: putting request to queue")
            request_queue.put(pipelinedata)

        return jsonify({}), 200
        
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
    Start the server on node2
    """
    global request_queue
    assert NODE_NUMBER == 2, "This script should be run on node2 only."

    print("="*60)
    print("NODE2 SERVER STARTING")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")

    # Start worker process
    for i in range(PROCESSES):
        t = multiprocessing.Process(target=worker, daemon=True, args=(request_queue,))
        t.start()
    print(f"Started {PROCESSES} worker processes")

    hostport = NODE_2_IP

    hostname = hostport.split(':')[0]
    port = int(hostport.split(':')[1]) if ':' in hostport else 8000

    print(f"\nStarting Flask server on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()

