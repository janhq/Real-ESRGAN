import torch
import cv2
from tqdm import tqdm
from rbdnet import RRDBNet
from realesrgan import RealESRGANer as OriginalRealESRGANer
from real_esrganer import RealESRGANer
import torch_tensorrt

torch.set_grad_enabled(False)


tile = torch.tensor([128])
input_size = (512, 512)
out_scale = torch.tensor([1.5])
n_runs = 20

upsampler = OriginalRealESRGANer(
    scale=4,
    model_path="./experiments/pretrained_models/RealESRGAN_x4plus.pth",
    dni_weight=None,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=int(tile),
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0
)
img = cv2.imread('./test.png')
img = cv2.resize(img, input_size)
res = upsampler.enhance(img, outscale=float(out_scale))[0]
for i in tqdm(range(n_runs), desc='Original pipeline'):
    res = upsampler.enhance(img, outscale=float(out_scale))[0]
cv2.imwrite('test_upscaled_org.png', res)


# Script the core model
core_model = upsampler.model
scripted_core_model = torch.jit.script(core_model)

upsampler = RealESRGANer(ts_model=scripted_core_model, half=True)
scripted_model = torch.jit.script(upsampler) # and the entire pipeline :D

img = cv2.imread('./test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, input_size)
img = torch.from_numpy(img).to('cuda')
res = upsampler.forward(img, out_scale=out_scale, tile=tile)
res = scripted_model.forward(img)
for i in tqdm(range(n_runs), desc='Warm up TS'):
    res = scripted_model.forward(img, out_scale=out_scale, tile=tile)
for i in tqdm(range(n_runs), desc='TS'):
    res = scripted_model.forward(img, out_scale=out_scale, tile=tile)
res = res.cpu().numpy()
res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
cv2.imwrite('test_upscaled_ts.png', res)


# Compile the core model with TensorRT
trt_ts_module = torch_tensorrt.compile(upsampler.model,
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 3, 128, 128],
            opt_shape=[1, 3, 512, 512],
            max_shape=[1, 3, 1024, 1024],
            dtype=torch.float16),
    ],
    workspace_size = 16 * (1 << 20), # 16MB
    enabled_precisions = {torch.half}, # Run with FP16
)

# Work arround to save full pipeline in the same scripted module
torch.jit.save(trt_ts_module, "/tmp/real_esrgan_trt.ts")
trt_ts_module = torch.jit.load("/tmp/real_esrgan_trt.ts")

upsampler.model = trt_ts_module
res = upsampler.forward(img, out_scale=out_scale, tile=tile)
for i in tqdm(range(n_runs), desc='Warm up TRT'):
    res = upsampler.forward(img, out_scale=out_scale, tile=tile)
for i in tqdm(range(n_runs), desc='TRT'):
    res = upsampler.forward(img, out_scale=out_scale, tile=tile)

res = res.cpu().numpy()
res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
cv2.imwrite('test_upscaled_trt.png', res)


# Script the entire pipeline again while including core TensorRT model
scripted_trt_model = torch.jit.script(upsampler)
res = scripted_trt_model.forward(img, out_scale=out_scale, tile=tile)
for i in tqdm(range(n_runs), desc='Warm up scripted TRT'):
    res = scripted_trt_model.forward(img, out_scale=out_scale, tile=tile)
for i in tqdm(range(n_runs), desc='Scripted TRT'):
    res = scripted_trt_model.forward(img, out_scale=out_scale, tile=tile)

res = res.cpu().numpy()
res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
cv2.imwrite('test_upscaled_scripted_trt.png', res)

torch.jit.save(scripted_trt_model, "real_esrgan_scripted_trt.ts") # save the TRT embedded Torchscript
