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
from queue import Empty
from multiprocessing import Queue
import numpy as np
import logging
import atexit

LOGGER = logging.getLogger(__file__)


class QueueAdapter:
    def __init__(self, inp_queue: Queue, dump_fname: str,):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            dump_fname (str): dump file name, if empty no dump will be done
        """

        self.all_done = False
        self.inp_queue = inp_queue
        self.dump_fname = dump_fname
        if len(self.dump_fname):
            self.f_out = open(dump_fname, "ab")
            atexit.register(self._cleanup)

    def _cleanup(self):
        self.f_out.close()

    def read(self, size: int) -> bytes:
        """
        Simple adapter which meets the vali.PyDecoder readable object interface.
        It takes chunks from queue and gives them to decoder.
        Empty bytearray put into queue serves as 'all done' flag.

        Args:
            size (int): requested read size

        Returns:
            bytes: compressed video bytes
        """

        while not self.all_done:
            try:
                chunk = self.inp_queue.get(timeout=0.1)

                if chunk is None:
                    self.all_done = True
                    return bytearray()

                if len(self.dump_fname):
                    self.f_out.write(chunk)

                return chunk

            except Empty:
                continue

            except ValueError:
                break

            except Exception as e:
                LOGGER.error(f"Unexpected excepton: {str(e)}")

        return bytearray()


class NvDecoder:
    def __init__(self,
                 inp_queue: Queue,
                 dump_fname: str,
                 dump_ext: str,
                 gpu_id=0,
                 async_depth=1):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            dump_fname (str): dump file name, if empty no dump will be done
            dump_ext (str): dump file dump_ext
            gpu_id (int, optional): GPU to run on. Defaults to 0.
            async_depth (int, optional): amount of available async tasks. Defaults to 1
        """

        # Generate dump filename with extension
        fname_plus_ext = dump_fname
        if len(fname_plus_ext):
            fname_plus_ext += "."
            fname_plus_ext += dump_ext

        # Adapter that allows decoder to read video track chunks from queue
        self.adapter = QueueAdapter(inp_queue, fname_plus_ext)

        # First try to create HW-accelerated decoder.
        # Some codecs / formats may not be supported, fall back to SW decoder.
        try:
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id)
        except Exception:
            # No exception handling here.
            # Failure to create SW decoder is fatal.
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id=-1)

        # Allocate surfaces
        self.async_depth = async_depth
        self.dec_pool = Queue(maxsize=self.async_depth)
        for i in range(0, self.async_depth):
            surf = vali.Surface.Make(
                self.py_dec.Format, self.py_dec.Width, self.py_dec.Height, gpu_id)
            self.dec_pool.put(surf)

        # SW decoder outputs to numpy array.
        # Have to initialize uploader to keep decoded frames in vRAM.
        if not self.py_dec.IsAccelerated:
            self.uploader = vali.PyFrameUploader(gpu_id)

    def width(self) -> int:
        """
        Get width

        Returns:
            int: decoded frame width in pixels
        """
        return self.py_dec.Width

    def height(self) -> int:
        """
        Get height

        Returns:
            int: decoded frame height in pixels
        """
        return self.py_dec.Height

    def format(self) -> vali.PixelFormat:
        """
        Get decoder pixel format

        Returns:
            vali.PixelFormat: decoder format
        """
        return self.py_dec.Format

    def async_depth(self) -> int:
        """
        Get amount of async decode tasks that can be run concurently in any
        given moment of time.

        Returns:
            int: decoder async depth
        """
        return self.async_depth

    async def unlock(self, surf: vali.Surface):
        """
        Returs surface back to decoder pool. Will not return until surface is
        put back to decoder pool. \\
        
        Use it together with :func:`NvDecoder.decode_lock`

        Args:
            surf (vali.Surface): surface to return.
        """
        self.dec_pool.put(surf)

    async def decode_lock(self) -> vali.Surface:
        """
        Decode single video frame. Use it together with :func:`NvDecoder.unlock` \\
        Blocks until there's available surface in decoder pool. \\
        Will return None upon EOF or decoding error.

        You can launch multiple tasks with asyncio. \\
        Amount of tasks to be run in any given moment of time is equal to
        async_depth.

        Returns:
            vali.Surface: Surface with reconstructed pixels.
        """

        try:
            surf = self.dec_pool.get()

            if self.py_dec.IsAccelerated:
                success, info = self.py_dec.DecodeSingleSurface(surf)

                if info == vali.TaskExecInfo.END_OF_STREAM:
                    return None

                if not success:
                    LOGGER.error(f"Failed to decode surface: {info}")
                    return None
            else:
                dec_frame = np.ndarray(shape=(self.py_dec.HostFrameSize),
                                       dtype=np.uint8)
                success, info = self.py_dec.DecodeSingleFrame(dec_frame)
                if not success:
                    LOGGER.error(f"Failed to decode frame: {info}")
                    return None

                success, info = self.uploader.Run(dec_frame, surf)
                if not success:
                    LOGGER.error(f"Failed to upload frame: {info}")
                    return None

            return surf

        except Exception as e:
            LOGGER.error(f"Unexpected exception: {str(e)}")
            return None
