import math
from typing import Optional, Tuple

import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn

torch.set_grad_enabled(False)

class RealESRGANer(nn.Module):
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        ts_model (torch.jit.ScriptModule): Scripted RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) model
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 ts_model: torch.jit.ScriptModule,
                 scale: int = 4,
                 half=False,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.scale = scale
        self.model = ts_model
        self.half = half
        if self.half:
            self.model = self.model.half()

    def pre_process(self, img: torch.Tensor, pre_pad: int) -> Tuple[torch.Tensor, Optional[int], int, int]:
        """ Pre-process, such as pre-pad and mod pad, so that the images can be divisible """
        img = img.to(torch.float32)
        img = img / 255
        img = img.permute(2, 0, 1).unsqueeze(0) #.to(self.device)
        if self.half:
            img = img.half()

        # pre_pad
        if pre_pad != 0:
            img = F.pad(img, (0, pre_pad, 0, pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        mod_pad_h, mod_pad_w = 0, 0
        if mod_scale is not None:
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                mod_pad_h = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                mod_pad_w = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return img, mod_scale, mod_pad_h, mod_pad_w

    def tile_process(self, img: torch.Tensor, tile_size: int, tile_pad: int) -> torch.Tensor:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.model(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        return output

    def post_process(self, output: torch.Tensor, mod_scale: Optional[int], out_scale: Optional[float], org_h: int, org_w: int, mod_pad_h: int, mod_pad_w: int, pre_pad: int) -> torch.Tensor:
        """ Post processing logic """
        # remove extra pad
        if mod_scale is not None:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * self.scale, 0:w - mod_pad_w * self.scale]
        # remove prepad
        if pre_pad != 0:
            _, _, h, w = output.size()
            print('Before remove prepad', output.shape)
            output = output[:, :, 0:h - pre_pad * self.scale, 0:w - pre_pad * self.scale]
            print('After remove prepad', output.shape)

        output = output.float().clamp_(0, 1)

        if out_scale is not None and out_scale != self.scale:
            out_h = int(org_h * out_scale)
            out_w = int(org_w * out_scale)
            output = F.interpolate(output, (out_h, out_w))

        output = output.squeeze().permute(1, 2, 0)
        output = (output * 255.0).round().to(torch.uint8)
        return output

    # def forward(self, img: torch.Tensor, out_scale: Optional[float] = None, tile: int = 0, tile_pad: int = 10, pre_pad: int = 0) -> torch.Tensor:
    #     """
    #     Enhance the input image

    #     Args:
    #         img (torch.Tensor): The RGB image to be enhanced
    #         out_scale (float): The output scale
    #         tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
    #             input images into tiles, and then process each of them. Finally, they will be merged into one image.
    #             0 denotes for do not use tile. Default: 0.
    #         tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
    #         pre_pad (int): Pad the input images to avoid border artifacts. Default: 0.
    #     """
    #     assert img.ndim == 3 and img.shape[-1] == 3, "Only support RGB image"
    #     org_h, org_w = img.shape[:2]
    #     img, mod_scale, mod_pad_h, mod_pad_w = self.pre_process(img, pre_pad=pre_pad)
    #     if tile > 0:
    #         output = self.tile_process(img, tile_size=tile, tile_pad=tile_pad)
    #     else:
    #         output = self.model(img)
    #     output = self.post_process(output, mod_scale=mod_scale, out_scale=out_scale, org_h=org_h, org_w=org_w, mod_pad_h=mod_pad_h, mod_pad_w=mod_pad_w, pre_pad=pre_pad)
    #     return output

    def make_img_shape_divisible_by_8(self, img: torch.Tensor) -> torch.Tensor:
        height, width = img.shape[:2]
        target_height = (height // 8 + 1) * 8 if height % 8 != 0 else height
        target_width = (width // 8 + 1) * 8 if width % 8 != 0 else width
        if target_height == height and target_width == width:
            return img
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, (target_height, target_width))
        img = img.squeeze().permute(1, 2, 0)
        img = (img * 255.0).round().to(torch.uint8)
        return img

    def forward(self,
                img: torch.Tensor,
                out_scale: torch.Tensor = torch.Tensor([4]).to(torch.float32),
                tile: torch.Tensor = torch.Tensor([0]).to(torch.int32),
                tile_pad: torch.Tensor = torch.Tensor([10]).to(torch.int32),
                pre_pad: torch.Tensor = torch.Tensor([0]).to(torch.int32)) -> torch.Tensor:
        assert img.ndim == 3 and img.shape[-1] == 3, "Only support RGB image"
        # assert out_scale.ndim == 1 and tile.ndim == 1 and tile_pad.ndim == 1 and pre_pad.ndim == 1

        out_scale = float(out_scale)
        tile = int(tile)
        tile_pad = int(tile_pad)
        pre_pad = int(pre_pad)
        img = self.make_img_shape_divisible_by_8(img)
        org_h, org_w = img.shape[:2]
        img, mod_scale, mod_pad_h, mod_pad_w = self.pre_process(img, pre_pad=pre_pad)
        if tile > 0:
            output = self.tile_process(img, tile_size=tile, tile_pad=tile_pad)
        else:
            output = self.model(img)
        output = self.post_process(output, mod_scale=mod_scale, out_scale=out_scale, org_h=org_h, org_w=org_w, mod_pad_h=mod_pad_h, mod_pad_w=mod_pad_w, pre_pad=pre_pad)
        return output