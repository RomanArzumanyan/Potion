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
from multiprocessing.synchronize import Event as SyncEvent
import atexit

logger = logging.getLogger(__file__)


class QueueAdapter:
    def __init__(self, inp_queue: Queue, dump: bool, stop_event: SyncEvent,):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            dump (bool): flag, set up to tell adapter to dump video track
            stop_event (SyncEvent): multiprocessing event which stops adapter from reading chunks from queue
        """

        self.inp_queue = inp_queue
        self.stop_event = stop_event
        self.dump = dump
        self.f_out = open("dump.bin", "ab")
        atexit.register(self.cleanup)

    def cleanup(self):
        self.f_out.close()

    def read(self, size: int) -> bytes:
        """
        Simple adapter which meets the vali.PyDecoder readable object interface.
        It takes chunks from queue and gives them to decoder.

        Args:
            size (int): requested read size

        Returns:
            bytes: compressed video bytes
        """

        while not self.stop_event.is_set():
            try:
                chunk = self.inp_queue.get_nowait()
                if self.dump:
                    self.f_out.write(chunk)
                return chunk

            except Empty:
                continue

            except ValueError:
                logger.info("Queue is closed.")
                return bytes()

            except Exception as e:
                logger.error(f"Unexpected excepton: {str(e)}")

        return bytes()


class NvDecoder:
    def __init__(self,
                 inp_queue: Queue,
                 stop_event: SyncEvent,
                 dump: bool,
                 gpu_id=0,):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            stop_event (SyncEvent): multiprocessing event which stops adapter from reading chunks from queue
            dump (bool): flag, set up to tell adapter to dump video track
            gpu_id (int, optional): GPU to run on. Defaults to 0.
        """

        self.adapter = QueueAdapter(inp_queue, dump, stop_event)

        # First try to create HW-accelerated decoder.
        # Some codecs / formats may not be supported, fall back to SW decoder then.
        try:
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id)
        except Exception as e:
            # No exception handling here.
            # Failure to create SW decoder is fatal.
            logger.warning(f"Failed to create HW decoder, reason: {str(e)}")
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id=-1)

        width = self.py_dec.Width
        height = self.py_dec.Height

        self.surfaces = [
            vali.Surface.Make(self.py_dec.Format, width, height, gpu_id),
            vali.Surface.Make(vali.PixelFormat.RGB, width, height, gpu_id),
            vali.Surface.Make(vali.PixelFormat.RGB_PLANAR,
                              width, height, gpu_id),
        ]

        self.convs = [
            vali.PySurfaceConverter(
                self.surfaces[0].Format, self.surfaces[1].Format, gpu_id),
            vali.PySurfaceConverter(
                self.surfaces[1].Format, self.surfaces[2].Format, gpu_id),
        ]

        # SW decoder outputs to numpy array.
        # Have to initialize uploader to keep decoded frames always in vRAM.
        if not self.py_dec.IsAccelerated:
            self.uploader = vali.PyFrameUploader(self.gpu_id)
            self.dec_frame = np.ndarray(shape=(self.py_dec.HostFrameSize),
                                        dtype=np.uint8)

    def decode(self) -> vali.Surface:
        """
        Decode single video frame

        Returns:
            vali.Surface: Surface with reconstructed pixels.
        """

        try:
            if self.py_dec.IsAccelerated:
                success, info = self.py_dec.DecodeSingleSurface(
                    self.surfaces[0])
                if not success:
                    logger.error(info)
                    return None
            else:
                success, info = self.py_dec.DecodeSingleFrame(self.dec_frame)
                if not success:
                    logger.error(info)
                    return None

                success, info = self.uploader.Run(
                    self.dec_frame, self.surfaces[0])
                if not success:
                    logger.error(info)
                    return None

                # Color conversion
            for i in range(0, len(self.convs)):
                success, info = self.convs[i].Run(
                    self.surfaces[i], self.surfaces[i + 1])
                if not success:
                    logger.error(info)
                    return None

            return self.surfaces[2]

        except Exception as e:
            logger.error(info)
            return None
