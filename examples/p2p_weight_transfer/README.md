# Tests of p2p rdma weight updating

- Test of single node, qwen3-4B : `python examples/p2p_weight_transfer/weight_transfer_with_rdma.py`
- Test of multiple nodes, qwen3-30B-A3B: 
```
# Step 1: prepare the ckpts
# on a single node
# ckpts will be saved in /root/multinode
bash examples/p2p_weight_transfer/prepare-qwen3-30B-A3B.sh


# Step 2: run the ray jobs
# Here is a 4-node test. Considering there's a HEAD_NODE_IP 
# on NODE 0 
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh rdma 0 $HEAD_NODE_IP 

# on NODE 1-3

bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh rdma 1 $HEAD_NODE_IP 
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh rdma 2 $HEAD_NODE_IP 
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh rdma 3 $HEAD_NODE_IP 
```