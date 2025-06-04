# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from queue import Empty, Queue as SimpleQueue

import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event as SyncEvent

import python_potion.decoder as decoder
import python_potion.converter as converter
import python_vali as vali
import time
import concurrent.futures

from argparse import Namespace
from enum import Enum
from ultralytics import YOLO
import torch
import json
import nvtx


LOGGER = logging.getLogger(__file__)


class ClientState(Enum):
    RUNNING = 0,
    EOF = 1,
    ERROR = 2


class InferenceClient():
    def __init__(self, flags: Namespace, inp_queue: Queue,):
        """
        Constructor.

        Args:
            flags (Namespace): parsed CLI args
            inp_queue (Queue): queue with video track chunks

        Raises:
            InferenceServerException: if triton throws an exception
        """

        self.gpu_id = flags.gpu_id
        self.flags = flags

        # Number of surfaces taken and number of responses received
        self.num_surf = 0
        self.num_resp = 0

        # Width and height expected by NN
        self.w = 640
        self.h = 640

        # Create decoder and converter
        self.dec = decoder.Decoder(inp_queue, self.flags)

        # Use YOLO model for segmentation and tracking
        self.model = YOLO("yolo11n.pt")

        params = {
            "src_fmt": self.dec.format(),
            "dst_fmt": vali.PixelFormat.RGB_32F_PLANAR,
            "dst_w": self.w,
            "dst_h": self.h,
        }

        self.conv = converter.Converter(
            params, self.flags, self.gpu_id, self.dec.cuda_stream())

        # Deal with batch size etc.
        self.batch_size = self.flags.batch_size
        self.supports_batching = self.batch_size > 0
        if not self.supports_batching and self.batch_size != 1:
            raise RuntimeError("ERROR: This model doesn't support batching.")

        # Async stuff
        self.tasks = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Results
        self.results = SimpleQueue()
        self.all_done = False

    @nvtx.annotate()
    def _process(self, results, request_id) -> None:
        """
        Process inference result and put it into stdout.
        """
        for detection in results:
            result = {request_id: detection}
            self.results.put(result)

    @nvtx.annotate()
    def _signal_end(self) -> None:
        """
        Put sentinel message into results queue.
        It signals that no more tasks will be submitted.
        """

        def _impl(results: SimpleQueue) -> None:
            results.put(None)

        future = self.executor.submit(_impl, self.results)
        self.tasks.add(future)
        future.add_done_callback(self.tasks.remove)

    @nvtx.annotate()
    def _inference(self, surf_inf: list[vali.Surface], request_id: str) -> None:
        """
        Run inference, get response
        """

        assert len(surf_inf) == self.batch_size

        try:
            img = torch.from_dlpack(surf_inf[0]).clamp(
                0.0, 1.0).clone().detach().reshape([1, 3, self.h, self.w])

            results = self.model.track(img, persist=True, verbose=False)
            self._process(json.loads(results[0].to_json()), request_id)
            self.num_resp += 1

        except Exception as e:
            LOGGER.error("Failed to run inference: " + str(e))

    @nvtx.annotate()
    def push_frame(self, buf_stop: SyncEvent, start_time: float) -> bool:
        # Signal stop
        if buf_stop is not None and self.flags.time > 0.0:
            if time.time() - start_time > self.flags.time:
                buf_stop.set()

        try:
            # Decode Surface
            surf_src = self.dec.decode()
            if surf_src is None:
                self._signal_end()
                return False

            # Process to match NN expectations
            surf_dst = self.conv.convert(surf_src)
            if surf_dst is None:
                self._signal_end()
                return False

            # Run inference
            future = self.executor.submit(
                self._inference, [surf_dst], str(self.num_surf))
            self.tasks.add(future)
            future.add_done_callback(self.tasks.remove)

            # Increase surfaces counter
            self.num_surf += 1

        except Exception as e:
            LOGGER.error(
                f"Frame {self.num_surf}. Unexpected excepton: {str(e)}")
            self._signal_end()
            return False

        return True

    def get_result(self) -> tuple[bool, dict]:
        """
        Get inference result

        Returns:
            tuple[bool, dict]:
                First element is True if there are more results, False otherwise
                Second element is dict with results, may be None if no responses are ready yet.
        """

        res = None

        if not self.all_done:
            try:
                res = self.results.get_nowait()
                if not res:
                    self.all_done = True
            except Empty:
                pass

        return (not self.all_done, res)
