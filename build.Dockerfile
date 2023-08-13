FROM nvcr.io/nvidia/tritonserver:23.06-py3
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install ipdb
ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace