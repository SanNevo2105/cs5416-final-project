## Advanced Implementation
The monolithic pipeline is broken down in to `node0.py`, `node1.py`, `node2.py`, and `service.py`. `service.py` is a library that defines the microservices. `node0.py`, `node1.py`, and `node2.py` starts the services on each node.

opportunistic batching is implemented.

## How to run

### 0. Create a virtual environment (Optional)
```
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
```

### 1. Install dependencies
```
./install.sh
```

### 2. Run nodes

In one terminal, run node 0
```
TOTAL_NODES=3 NODE_NUMBER=0 NODE_0_IP=127.0.0.1:5000 NODE_1_IP=127.0.0.1:5001 NODE_2_IP=127.0.0.1:5002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh
```

In another terminal, run node 1
```
TOTAL_NODES=3 NODE_NUMBER=1 NODE_0_IP=127.0.0.1:5000 NODE_1_IP=127.0.0.1:5001 NODE_2_IP=127.0.0.1:5002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh
```

In another terminal, run node 2
```
TOTAL_NODES=3 NODE_NUMBER=2 NODE_0_IP=127.0.0.1:5000 NODE_1_IP=127.0.0.1:5001 NODE_2_IP=127.0.0.1:5002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh
```

### 3. Run benchmark client
Run `client.py` to see if everything is working as expected
```
NODE_0_IP=127.0.0.1:5000 python3 client.py
```

Run `bench_client.py` to benchmark the distributed pipeline. See `bench_client.py` for more running options.
```
NODE_0_IP=127.0.0.1:5000 \python3 bench_client.py --num-requests 100 --concurrency 8 --label "batch_size=4"
```

## Potential Experiments
### Jaxin TODO
1. Change BATCH_SIZE, BATCH_WAIT_SECONDS and see how it affects throughput and latency.

### Nipat TODO
1. How much memory does each stage use? What is the throughput and latency for each stage?

2. Currently node 0 does embedding, node 1 does FAISS, document retrieval, and reranking, and node 2 does LLM generation, sentiment analysis, and safety filtering. node 0 finishes in about 0.2s, node 1 finishes in about 5s, but node 2 finishes in about 80s for a batch of 2. Seems like node 2 is the bottleneck. Is there a way to improve this? What is the best way to split the stages across the 3 nodes? Maybe experiment 1 above can help you make the decision.

3. Currently each node has 3 threads running. Maybe try changing the number of threads per node and see how it affects throughput and latency? Our code will be tested on a CPU with 16 cores.

**Don't forget to record the experiment data as you will need to write them in the report!**
