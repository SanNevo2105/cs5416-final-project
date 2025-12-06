## Advanced Implementation
The monolithic pipeline is broken down in to `node0.py`, `node1.py`, `node2.py`, and `service.py`. `service.py` is a library that defines the microservices. `node0.py`, `node1.py`, and `node2.py` starts the services on each node.

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
