import random
import time
from uuid import uuid4
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tqdm import tqdm

url = "0.0.0.0:9001"
model_name = "real_esrgan"
num_runs = 100

img_path = './test.png'
img = np.asarray(Image.open(img_path).convert('RGB'))

outscale = 4
tile = 0
tile_pad = 0
pre_pad = 0

triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)

all_results = []

def callback(results, start_time, result, error):
    images = result.as_numpy("out__0")
    # print(images.shape)
    # if images.ndim == 3:
    #     images = images[None, ...]
    # Image.fromarray(images[0]).save('images.jpg')
    results.append(
        {
            'output': None,
            'error': error,
            'start_time': start_time,
            'tooks': time.time() - start_time,
        }
    )


for i in tqdm(range(num_runs), desc="Submit requests"):
    img_in = grpcclient.InferInput(name="img__0", shape=img.shape, datatype="UINT8")
    out_scale_in = grpcclient.InferInput(name="outscale__1", shape=(1,), datatype="FP32")
    tile_in = grpcclient.InferInput("tile__2", (1,), "INT32")
    tile_pad_in = grpcclient.InferInput("tile_pad__3", (1,), "INT32")
    pre_pad_in = grpcclient.InferInput("pre_pad__4", (1,), "INT32")

    images = grpcclient.InferRequestedOutput(name="out__0")

    img_in.set_data_from_numpy(img)
    out_scale_in.set_data_from_numpy(np.asarray([outscale], dtype=np.float32))
    tile_in.set_data_from_numpy(np.asarray([tile], dtype=np.int32))
    tile_pad_in.set_data_from_numpy(np.asarray([tile_pad], dtype=np.int32))
    pre_pad_in.set_data_from_numpy(np.asarray([pre_pad], dtype=np.int32))

    triton_client.async_infer(
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
        callback=partial(callback, results=all_results, start_time=time.time()),
    )

start = time.time()
time_out = 20 * num_runs  # secs
pbar = tqdm(total=num_runs, desc='Waiting')
computed_so_far = len(all_results)
while time_out > 0:
    curr_computed = len(all_results)
    newly_computed = curr_computed - computed_so_far
    computed_so_far = curr_computed
    pbar.update(newly_computed)
    time_out -= 1
    if curr_computed >= num_runs:
        break
    time.sleep(0.5)
pbar.close()
end = time.time()

print(f"On average, inference took {((end-start)/num_runs) * 1000} ms")
# tooks = [each['tooks'] * 1000 for each in all_results]
# print(f"Queue + process time distribution: {tooks}")

# import json
# with open('grpc_telemetry.json', 'w') as f:
#     json.dump(all_results, f)
