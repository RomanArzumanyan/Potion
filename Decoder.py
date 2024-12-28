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
        self.inp_queue = inp_queue
        self.stop_event = stop_event
        self.dump = dump
        self.f_out = open("dump.bin", "ab")
        atexit.register(self.cleanup)

    def cleanup(self):
        self.f_out.close()

    def read(self, size: int) -> bytes:
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

        self.adapter = QueueAdapter(inp_queue, dump, stop_event)
        self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id)

        width = self.py_dec.Width
        height = self.py_dec.Height

        self.surfaces = [
            vali.Surface.Make(vali.PixelFormat.NV12, width, height, gpu_id),
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

    def decode(self) -> vali.Surface:
        try:
            success, info = self.py_dec.DecodeSingleSurface(self.surfaces[0])
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
