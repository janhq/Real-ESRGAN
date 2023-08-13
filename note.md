- Download the orignal pytorch weight:
```
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O experiments/pretrained_models/RealESRGAN_x4plus.pth
```
- Build the conversion docker image:
```
DOCKER_BUILDKIT=1 docker build -f build.Dockerfile -t janresearch.azurecr.io/real_esrgan_build:triton23.06-trt8.6 .
```
- Run the docker image up to do the conversion:
```
docker run --gpus device=1 --rm -it -v $PWD:/workspace janresearch.azurecr.io/real_esrgan_build:triton23.06-trt8.6
python3 convert.py
```
- Exit the container and copy the exported model to the triton repo:
```
cp real_esrgan_scripted_trt.ts model_repo/real_esrgan/1/model.pt
```
- Run the triton server up for serving
```
docker run --gpus device=1 --rm -it \
       -p 9000:8000 -p 9001:8001 -p 9002:8002 \
       -v $PWD/model_repo:/models \
       janresearch.azurecr.io/real_esrgan_build:triton23.06-trt8.6 \
       tritonserver --model-repository=/models
```
- Infer test:
```
python3 client_grpc_async.py
```