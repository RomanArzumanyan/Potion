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

        self.flags = flags

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

        self._build_chain(self.src_fmt, self.dst_fmt)

    def _build_chain(self, src: vali.PixelFormat, dst: vali.PixelFormat):
        """
        Build color conversion chain

        Args:
            src (vali.PixelFormat): input pixel format
            dst (vali.PixelFormat): NN target pixel format

        Raises:
            RuntimeError: if chain can't be built
        """

        # Get list of supported conversion and generate adjacency matrix
        convs = {}
        for conv in vali.PySurfaceConverter.Conversions():
            key, val = conv[0], conv[1]
            if not key in convs.keys():
                convs[key] = []
            convs[key].append(val)

        # Find shortest path in graph by given adj matrix
        chain = common.find_shortest_path(convs, src, dst)
        if len(chain) < 2:
            raise RuntimeError(
                f"Can't build color conversion chain from {src} to {dst}")

        # Create converters
        self.conv = []
        for i in range(0, len(chain) - 1):
            self.conv.append(vali.PySurfaceConverter(
                chain[i], chain[i+1], self.flags.gpu_id)
            )

        # Create surfaces and resizer
        self.surf = []
        self.need_resize = self.src_w != self.dst_w or self.src_h != self.dst_h
        start_idx = 1 - int(self.need_resize)
        for i in range(start_idx, len(chain)):
            self.surf.append(vali.Surface.Make(
                chain[i], self.dst_w, self.dst_h, self.flags.gpu_id))

        if self.need_resize:
            self.resz = vali.PySurfaceResizer(self.src_fmt, self.flags.gpu_id)

    def req_params(self) -> list[str]:
        """
        Get list of required converter parameters.

        Returns:
            list[str]: list of parameters
        """
        return self.req_params

    @nvtx.annotate()
    def convert(self, surf_src: vali.Surface, sync=False) -> vali.Surface:
        """
        Runs color conversion and resize if necessary. \
        All operations are run in async fashion without and CUDA Events being record. \
        This is done on purpose, since a blocking DtoH CUDA memcpy call shall be done to read
        Surface into RAM and send for inference. 

        Args:
            surf_src (vali.Surface): input surface
            sync (Bool): if True, will record and sync on last conversion, otherwise
            won't record and sync on any operations

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
        event = None
        for i in range(0, len(self.conv)):
            record = sync and i == len(self.conv) - 1
            success, info, event = self.conv[i].RunAsync(
                src=self.surf[i], dst=self.surf[i+1], record_event=record)

            if not success:
                LOGGER.error(f"Failed to convert surface: {info}")
                return None

        if event:
            event.Wait()

        return self.surf[-1]
