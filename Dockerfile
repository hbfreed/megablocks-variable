FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

RUN pip install stanford-stk==0.7.1

RUN pip install flash-attn

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks
