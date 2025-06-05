# Copyright 2024 Roman Arzumanyan.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http: // www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import python_vali as vali
import logging
from typing import Dict
from argparse import Namespace
import python_potion.common as common

import nvtx

LOGGER = logging.getLogger(__file__)


class Converter:
    """
    Use this class for color / data type conversion and resize.
    It owns the converted Surface, clone it if needed.
    """

    def __init__(self, params: Dict, flags: Namespace, gpu_id: int, cuda_stream: int):
        self.req_par = [
            "src_fmt",
            "dst_fmt",
            "dst_w",
            "dst_h",
        ]

        for param in self.req_par:
            if not param in params.keys():
                raise RuntimeError(
                    f"Parameter {param} not found. Required params: {self.req_par}")

        self.flags = flags

        self.src_fmt = params["src_fmt"]
        self.dst_fmt = params["dst_fmt"]

        self.dst_w = params["dst_w"]
        self.dst_h = params["dst_h"]

        # Only (semi-)planar yuv420 input is supported.
        fmts = [vali.PixelFormat.NV12, vali.PixelFormat.YUV420]
        if not self.src_fmt in fmts:
           raise RuntimeError(f"Unsupported input format {self.src_fmt}\n"
                              f"Supported formats: {fmts}")

        # Only RGB output is supported.
        fmts = [
            vali.PixelFormat.RGB,
            vali.PixelFormat.RGB_32F,
            vali.PixelFormat.RGB_PLANAR,
            vali.PixelFormat.RGB_32F_PLANAR,
        ]
        if not self.dst_fmt in fmts:
           raise RuntimeError(f"Unsupported output format {self.dst_fmt}\n"
                              f"Supported formats: {fmts}")

        # Combined rescaler-resizer
        self.py_ud = vali.PySurfaceUD(gpu_id, cuda_stream)

        # Inference surface
        self.surf_inf = vali.Surface.Make(
            self.dst_fmt, self.dst_w, self.dst_h, gpu_id)

        # OSD surface
        self.surf_osd = vali.Surface.Make(
            vali.PixelFormat.RGB, self.dst_w, self.dst_h, gpu_id)

    def req_params(self) -> list[str]:
        """
        Get required parameters list
        """
        return self.req_params

    @nvtx.annotate()
    def convert(self, surf_src: vali.Surface, sync=False) -> vali.Surface:
        """
        Resample and colorspace conversion
        """
        if surf_src.Format != self.src_fmt:
            raise RuntimeError("Input surface format mismatch")

        for surf_dst in [self.surf_inf, self.surf_osd]:
            success, info = self.py_ud.Run(
                surf_src, surf_dst) if sync else self.py_ud.Run(surf_src, surf_dst)
            if not success:
                LOGGER.error(f"Failed to resample-rescale surface: {info}")
                return None

        return self.surf_inf, self.surf_osd
