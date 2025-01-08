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

LOGGER = logging.getLogger(__file__)


class Converter:
    def __init__(self, params: Dict, gpu_id: int):
        """
        Constructor

        Args:
            params (Dict): dictionary with parameters
            gpu_id (int): GPU id to run on

        Raises:
            RuntimeError: if input or output formats aren't supported
        """

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
                              self.dst_w, self.dst_h, gpu_id)
        ]

        self.need_resize = self.src_w != self.dst_w or self.src_h != self.dst_h
        if self.need_resize:
            # Resize input Surface to decrease amount of pixels to be further processed
            self.resz = vali.PySurfaceResizer(self.src_fmt, gpu_id)
            self.surf.insert(0, vali.Surface.Make(
                self.src_fmt, self.dst_w, self.dst_h, gpu_id))

        # Converters
        self.conv = [
            vali.PySurfaceConverter(
                self.src_fmt, vali.PixelFormat.RGB, gpu_id),

            vali.PySurfaceConverter(
                vali.PixelFormat.RGB, vali.PixelFormat.RGB_32F, gpu_id),
        ]

        if self.dst_fmt == vali.PixelFormat.RGB_32F_PLANAR:
            self.surf.append(
                vali.Surface.Make(
                    vali.PixelFormat.RGB_32F, self.dst_w, self.dst_h, gpu_id)
            )

            self.conv.append(
                vali.PySurfaceConverter(
                    vali.PixelFormat.RGB_32F, vali.PixelFormat.RGB_32F_PLANAR, gpu_id)
            )

    def run(self, surf_src: vali.Surface, surf_dst: vali.Surface) -> None:
        """
        Runs color conversion and resize if necessary

        Args:
            surf_src (vali.Surface): input surface
            surf_dst (vali.Surface): output surface

        Raises:
            RuntimeError: in case of size / format mismatch
        """

        if surf_dst.Width != self.dst_w or surf_dst.Height != self.dst_h:
            raise RuntimeError("Output surface size mismatch")

        if surf_src.Width != self.src_w or surf_src.Height != self.src_h:
            raise RuntimeError("Input surface size mismatch")

        if surf_dst.Format != self.dst_fmt:
            raise RuntimeError("Output surface format mismatch")

        if surf_src.Format != self.src_fmt:
            raise RuntimeError("Input surface format mismatch")

        # Important:
        # Input and output surfaces will be temporarily insert into list
        # to avoid indices joggling.

        # Resize
        if self.need_resize:
            self.surf.insert(0, surf_src)
            success, info = self.resz.Run(self.surf[0], self.surf[1])
            self.surf.pop(0)
            if not success:
                LOGGER.error(f"Failed to resize surface: {info}")

        # Color conversion
        self.surf.append(surf_dst)
        for i in range(0, len(self.conv)):
            success, info = self.conv[i].Run(self.surf[i], self.surf[i+1])
            if not success:
                LOGGER.error(f"Failed to convert surface: {info}")
        self.surf.pop()
