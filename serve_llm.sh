#!/bin/bash

#set devices as per available gpu resources. We use 2 A100 80GB GPUs for this demo, so we set CUDA_VISIBLE_DEVICES to 5 and 6. Adjust as per your setup.

CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
--model ~/hf_models/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80 \
--served-model-name gemma-12b \
--tensor-parallel-size 1 \
--dtype bfloat16 \
--trust-remote-code \
--port 8001 > gemma12.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
--model ~/hf_models/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a \
--served-model-name gemma-27b \
--tensor-parallel-size 1 \
--dtype bfloat16 \
--trust-remote-code \
--port 8000 > gemma27.log 2>&1 &

wait