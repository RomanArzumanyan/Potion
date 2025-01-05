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

LOGGER = logging.getLogger(__file__)


class QueueAdapter:
    def __init__(self, inp_queue: Queue, stop_event: SyncEvent, dump_fname: str,):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            stop_event (SyncEvent): multiprocessing event which stops adapter from reading chunks from queue
            dump_fname (str): dump file name, if empty no dump will be done
        """

        self.inp_queue = inp_queue
        self.stop_event = stop_event
        self.dump_fname = dump_fname
        if len(self.dump_fname):
            self.f_out = open(dump_fname, "ab")
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
                if len(self.dump_fname):
                    self.f_out.write(chunk)
                return chunk

            except Empty:
                continue

            except ValueError:
                LOGGER.info("Queue is closed.")
                return bytes()

            except Exception as e:
                LOGGER.error(f"Unexpected excepton: {str(e)}")

        return bytes()


class NvDecoder:
    def __init__(self,
                 inp_queue: Queue,
                 stop_event: SyncEvent,
                 dump_fname: str,
                 dump_ext: str,
                 gpu_id=0,):
        """
        Constructor

        Args:
            inp_queue (Queue): queue with video chunks
            stop_event (SyncEvent): multiprocessing event which stops adapter from reading chunks from queue
            dump_fname (str): dump file name, if empty no dump will be done
            dump_ext (str): dump file dump_ext
            gpu_id (int, optional): GPU to run on. Defaults to 0.
        """

        fname_plus_ext = dump_fname
        if len(fname_plus_ext):
            fname_plus_ext += "."
            fname_plus_ext += dump_ext
        self.adapter = QueueAdapter(inp_queue, stop_event, fname_plus_ext)

        # First try to create HW-accelerated decoder.
        # Some codecs / formats may not be supported, fall back to SW decoder then.
        try:
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id)
        except Exception as e:
            # No exception handling here.
            # Failure to create SW decoder is fatal.
            LOGGER.warning(f"Failed to create HW decoder, reason: {str(e)}")
            self.py_dec = vali.PyDecoder(self.adapter, {}, gpu_id=-1)

        self.surf = vali.Surface.Make(
            self.py_dec.Format, self.py_dec.Width, self.py_dec.Height, gpu_id)

        # SW decoder outputs to numpy array.
        # Have to initialize uploader to keep decoded frames always in vRAM.
        if not self.py_dec.IsAccelerated:
            self.uploader = vali.PyFrameUploader(gpu_id)
            self.dec_frame = np.ndarray(shape=(self.py_dec.HostFrameSize),
                                        dtype=np.uint8)

    def decode(self) -> vali.Surface:
        """
        Decode single video frame

        Returns:
            vali.Surface: Surface with reconstructed pixels.
        """

        try:
            pkt_data = vali.PacketData()
            if self.py_dec.IsAccelerated:
                success, info = self.py_dec.DecodeSingleSurface(
                    self.surf, pkt_data)
                if not success:
                    LOGGER.error(info)
                    return None
            else:
                success, info = self.py_dec.DecodeSingleFrame(
                    self.dec_frame, pkt_data)
                if not success:
                    LOGGER.error(info)
                    return None

                success, info = self.uploader.Run(
                    self.dec_frame, self.surf)
                if not success:
                    LOGGER.error(info)
                    return None

            return self.surf

        except Exception as e:
            LOGGER.error(info)
            return None
