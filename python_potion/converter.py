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

import nvtx

LOGGER = logging.getLogger(__file__)


class Converter:
    """
    Use this class for color / data type conversion and resize.
    It owns the converted Surface, clone it if needed.
    """

    def __init__(self, params: Dict, flags: Namespace):
        """
        Constructor

        Args:
            params (Dict): dictionary with parameters
            flags (Namespace): parsed CLI args

        Raises:
            RuntimeError: if input or output formats aren't supported or one of
            requested parameters is missing
        """
        self.req_par = [
            "src_fmt",
            "dst_fmt",
            "src_w",
            "src_h",
            "dst_w",
            "dst_h"
        ]

        for param in self.req_par:
            if not param in params.keys():
                raise RuntimeError(
                    f"Parameter {param} not found. Required params: {self.req_par}")

        self.src_fmt = params["src_fmt"]
        self.dst_fmt = params["dst_fmt"]

        self.src_w = params["src_w"]
        self.src_h = params["src_h"]
        self.dst_w = params["dst_w"]
        self.dst_h = params["dst_h"]

        # Only (semi-)planar yuv420 input is supported.
        fmts = [vali.PixelFormat.NV12, vali.PixelFormat.YUV420]
        if not self.src_fmt in fmts:
           raise RuntimeError(f"Unsupported input format {self.src_fmt}\n"
                              f"Supported formats: {fmts}")

        # Only packed / planar float32 output is supported.
        fmts = [vali.PixelFormat.RGB_32F, vali.PixelFormat.RGB_32F_PLANAR]
        if not self.dst_fmt in fmts:
           raise RuntimeError(f"Unsupported output format {self.dst_fmt}\n"
                              f"Supported formats: {fmts}")

        # Surfaces for conversion chain
        self.surf = [
            vali.Surface.Make(vali.PixelFormat.RGB,
                              self.dst_w, self.dst_h, flags.gpu_id)
        ]

        self.need_resize = self.src_w != self.dst_w or self.src_h != self.dst_h
        if self.need_resize:
            # Resize input Surface to decrease amount of pixels to be further processed
            self.resz = vali.PySurfaceResizer(self.src_fmt, flags.gpu_id)
            self.surf.insert(0, vali.Surface.Make(
                self.src_fmt, self.dst_w, self.dst_h, flags.gpu_id))

        # Converters
        self.conv = [
            vali.PySurfaceConverter(
                self.src_fmt, vali.PixelFormat.RGB, flags.gpu_id),

            vali.PySurfaceConverter(
                vali.PixelFormat.RGB, vali.PixelFormat.RGB_32F, flags.gpu_id),
        ]

        if self.dst_fmt == vali.PixelFormat.RGB_32F_PLANAR:
            self.surf.append(
                vali.Surface.Make(
                    vali.PixelFormat.RGB_32F, self.dst_w, self.dst_h, flags.gpu_id)
            )

            self.conv.append(
                vali.PySurfaceConverter(
                    vali.PixelFormat.RGB_32F, vali.PixelFormat.RGB_32F_PLANAR, flags.gpu_id)
            )

        self.surf.append(
            vali.Surface.Make(self.dst_fmt, self.dst_w,
                              self.dst_h, flags.gpu_id)
        )

    def req_params(self) -> list[str]:
        """
        Get list of required converter parameters.

        Returns:
            list[str]: list of parameters
        """
        return self.req_params

    @nvtx.annotate()
    def convert(self, surf_src: vali.Surface) -> vali.Surface:
        """
        Runs color conversion and resize if necessary. \
        All operations are run in async fashion without and CUDA Events being record. \
        This is done on purpose, since a blocking DtoH CUDA memcpy call shall be done to read
        Surface into RAM and send for inference. 

        Args:
            surf_src (vali.Surface): input surface

        Returns:
            vali.Surface: Surface with converted pixels.

        Raises:
            RuntimeError: in case of size / format mismatch
        """

        if surf_src.Width != self.src_w or surf_src.Height != self.src_h:
            raise RuntimeError("Input surface size mismatch")

        if surf_src.Format != self.src_fmt:
            raise RuntimeError("Input surface format mismatch")

        # Resize
        if self.need_resize:
            success, info, _ = self.resz.RunAsync(
                src=surf_src, dst=self.surf[0], record_event=False)
            if not success:
                LOGGER.error(f"Failed to resize surface: {info}")
                return None

        # Color conversion.
        for i in range(0, len(self.conv)):
            success, info, _ = self.conv[i].RunAsync(
                src=self.surf[i], dst=self.surf[i+1], record_event=False)

            if not success:
                LOGGER.error(f"Failed to convert surface: {info}")
                return None

        return self.surf[-1]
