import time
from uuid import uuid4

import numpy as np
import tritonclient.http
from tqdm import tqdm
from PIL import Image

url = "0.0.0.0:9000"
model_name = "real_esrgan"
num_runs = 100

img_path = './test.png'
img = np.asarray(Image.open(img_path).convert('RGB'))
outscale = 4
tile = 0
tile_pad = 0
pre_pad = 0

triton_client = tritonclient.http.InferenceServerClient(
    url=url, verbose=False, concurrency=num_runs, network_timeout=3000,
)


async_requests = []
for i in tqdm(range(num_runs), desc="Submit requests"):
    img_in = tritonclient.http.InferInput(name="img__0", shape=img.shape, datatype="UINT8")
    out_scale_in = tritonclient.http.InferInput(name="outscale__1", shape=(1,), datatype="FP32")
    tile_in = tritonclient.http.InferInput("tile__2", (1,), "INT32")
    tile_pad_in = tritonclient.http.InferInput("tile_pad__3", (1,), "INT32")
    pre_pad_in = tritonclient.http.InferInput("pre_pad__4", (1,), "INT32")

    images = tritonclient.http.InferRequestedOutput(name="out__0", binary_data=False)

    img_in.set_data_from_numpy(img)
    out_scale_in.set_data_from_numpy(np.asarray([outscale], dtype=np.float32))
    tile_in.set_data_from_numpy(np.asarray([tile], dtype=np.int32))
    tile_pad_in.set_data_from_numpy(np.asarray([tile_pad], dtype=np.int32))
    pre_pad_in.set_data_from_numpy(np.asarray([pre_pad], dtype=np.int32))

    async_request = triton_client.async_infer(
        model_name,
        inputs=[
            img_in,
            out_scale_in,
            tile_in,
            tile_pad_in,
            pre_pad_in
        ],
        request_id=str(uuid4()),
        model_version="",
        outputs=[images],
    )
    async_requests.append(async_request)

tooks = []
for async_request in tqdm(async_requests, desc="Waiting"):
    st = time.time()
    res = async_request.get_result()
    images = res.as_numpy("out__0")
    end = time.time() - st
    tooks.append(end)

    # print(images.shape)
    # if images.ndim == 3:
    #     images = images[None, ...]
    # Image.fromarray(images[0]).save('images.jpg')

print(f"On average, inference took {np.mean(tooks)*1000} ms")
