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
from multiprocessing import Queue

LOGGER = logging.getLogger(__file__)


class SurfaceStorage:
    def __init__(self, surf: list[vali.Surface], max_size: int):
        """
        Create pool of surfaces list of given size.

        Args:
            surf (list[vali.Surface]): surfaces to copy / paste into pool
            max_size (int): pool size
        """
        self.pool = Queue(maxsize=max_size)
        self.size = max_size

        for i in range(0, max_size):
            copy = []
            copy.append(surf.Clone() for surf in surf)
            self.pool.put(copy)

    def get_lock(self) -> list[vali.Surface]:
        """
        Get surfaces from pool
        Use together with :func:`SurfaceStorage.put_unlock`

        Returns:
            list[vali.Surface]: Surfaces for use.
        """
        return self.pool.get()

    def put_unlock(self, surf: list[vali.Surface]) -> None:
        """
        Give surfaces back to pool.
        Use together with :func:`SurfaceStorage.get_lock`

        Args:
            surf (list[vali.Surface]): surfaces to put back into pool
        """
        self.pool.put(surf)


class Converter:
    def __init__(self, params: Dict, gpu_id=0, async_dept=1):
        """
        Constructor

        Args:
            params (Dict): dictionary with parameters
            gpu_id (int, optional): GPU to run on. Defaults to 0
            async_depth (int, optional): amount of available async tasks. Defaults to 1

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
        surf = [
            vali.Surface.Make(vali.PixelFormat.RGB,
                              self.dst_w, self.dst_h, gpu_id)
        ]

        self.need_resize = self.src_w != self.dst_w or self.src_h != self.dst_h
        if self.need_resize:
            # Resize input Surface to decrease amount of pixels to be further processed
            self.resz = vali.PySurfaceResizer(self.src_fmt, gpu_id)
            surf.insert(0, vali.Surface.Make(
                self.src_fmt, self.dst_w, self.dst_h, gpu_id))

        # Converters
        self.conv = [
            vali.PySurfaceConverter(
                self.src_fmt, vali.PixelFormat.RGB, gpu_id),

            vali.PySurfaceConverter(
                vali.PixelFormat.RGB, vali.PixelFormat.RGB_32F, gpu_id),
        ]

        if self.dst_fmt == vali.PixelFormat.RGB_32F_PLANAR:
            surf.append(
                vali.Surface.Make(
                    vali.PixelFormat.RGB_32F, self.dst_w, self.dst_h, gpu_id)
            )

            self.conv.append(
                vali.PySurfaceConverter(
                    vali.PixelFormat.RGB_32F, vali.PixelFormat.RGB_32F_PLANAR, gpu_id)
            )

        surf.append(vali.Surface.Make(
            self.dst_fmt, self.dst_w, self.dst_h, gpu_id))
        self.pool = SurfaceStorage(surf, async_dept)

    def async_depth(self) -> int:
        """
        Get amount of async tasks that can be run concurently in any
        given moment of time.

        Returns:
            int: converter async depth
        """
        return self.pool.size

    async def unlock(self, surf: list[vali.Surface]) -> None:
        """
        Put surfaces back to pool

        Args:
            surf (list[vali.Surface]): list of surfaces to return
        """
        self.pool.put_unlock(surf)

    async def cvt_lock(self, surf_src: vali.Surface,) -> list[vali.Surface]:
        """
        Runs color conversion and resize if necessary.
        Use it together with :func:`Converter.unlock`.

        Args:
            surf_src (vali.Surface): input surface

        Raises:
            RuntimeError: in case of size / format mismatch

        Returns:
            list[vali.Surface]: list of surface, you need the last one
        """

        if surf_src.Width != self.src_w or surf_src.Height != self.src_h:
            raise RuntimeError("Input surface size mismatch")

        if surf_src.Format != self.src_fmt:
            raise RuntimeError("Input surface format mismatch")

        # Take next available surface list from pool
        surf = self.pool.get_lock()

        # Resize
        if self.need_resize:
            success, info = self.resz.Run(surf_src, surf[0])
            if not success:
                LOGGER.error(f"Failed to resize surface: {info}")
                return None

        # Color conversion
        for i in range(0, len(self.conv)):
            success, info = self.conv[i].Run(surf[i], surf[i+1])
            if not success:
                LOGGER.error(f"Failed to convert surface: {info}")
                return None

        return surf
