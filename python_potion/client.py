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
import python_potion.osd as osd
import python_vali as vali
import time
import concurrent.futures

from argparse import Namespace
from enum import Enum
from ultralytics import YOLO
import torch
import json
import nvtx
import numpy as np

LOGGER = logging.getLogger(__file__)


class ClientState(Enum):
    RUNNING = 0,
    EOF = 1,
    ERROR = 2


class InferenceClient():
    def __init__(
            self,
            flags: Namespace,
            inp_queue: Queue,):
        """
        Constructor
        """

        self.gpu_id = flags.gpu_id
        self.flags = flags

        # Number of surfaces taken and number of responses received
        self.num_surf = 0
        self.num_resp = 0

        # Decoder, converter, OSD composer
        self.dec = decoder.Decoder(inp_queue, self.flags)

        # Width and height expected by NN
        self.w = 640
        self.h = 640

        # Width and height for OSD operations
        self.osd_w = self.w
        self.osd_h = int(float(self.w) / float(self.dec.width())
                         * float(self.dec.height()))

        params = {
            "src_fmt": self.dec.format(),
            "dst_fmt": vali.PixelFormat.RGB_32F_PLANAR,
            "dst_w": self.w,
            "dst_h": self.h,
            "osd_w": self.osd_w,
            "osd_h": self.osd_h,            
        }

        self.conv = converter.Converter(
            params, self.flags, self.gpu_id, self.dec.cuda_stream())

        self.osd = osd.Composer(flags)

        # Use YOLO model for segmentation and tracking
        self.model = YOLO("yolo11n.pt")

        # Async stuff
        self.push_done = False
        self.tasks = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @nvtx.annotate()
    def _osd(
            self,
            surf_inf: vali.Surface,
            surf_osd: vali.Surface,
            req_body: dict) -> np.ndarray:
        """
        Do OSD, compress into jpeg, insert into requet body
        """
        try:
            self.osd.draw(surf_inf, surf_osd, req_body)
            return self.osd.compress_jpeg(surf_osd)

        except Exception as e:
            LOGGER.error(f"Failed to do OSD: {str(e)}")

    @nvtx.annotate()
    def _process(
            self,
            surf_inf: vali.Surface,
            surf_osd: vali.Surface,
            results: list,
            request_id: str) -> None:
        """
        Process result, publish to Kafka topic
        """
        try:
            req_body = {}
            req_body["request_id"] = request_id
            req_body['detections'] = []

            for res in results:
                if res['confidence'] >= 0.5:
                    req_body['detections'].append(res)

            if not len(req_body['detections']):
                return

            jpeg = self._osd(surf_inf, surf_osd, req_body)
            with open(f"{request_id}.jpg", "wb") as f_out:
                f_out.write(jpeg)

        except Exception as e:
            LOGGER.error("Failed to publish: " + str(e))

    @nvtx.annotate()
    def _inference(
            self,
            surf_inf: vali.Surface,
            surf_osd: vali.Surface,
            request_id: str) -> None:
        """
        Run inference
        """
        try:
            img = torch.from_dlpack(surf_inf).clamp(
                0.0, 1.0).clone().detach().reshape([1, 3, self.h, self.w])

            results = self.model.track(img, persist=True, verbose=False)
            self._process(
                surf_inf.Clone(),
                surf_osd.Clone(),
                json.loads(results[0].to_json()),
                request_id)

            self.num_resp += 1

        except Exception as e:
            LOGGER.error("Failed to run inference: " + str(e))

    @nvtx.annotate()
    def _signal_push_end(self):
        self.push_done = True

    @nvtx.annotate()
    def push_frame(
            self,
            buf_stop: SyncEvent,
            start_time: float) -> bool:
        """
        Push single frame though processing chain
        """
        # Signal stop
        if buf_stop is not None and self.flags.time > 0.0:
            if time.time() - start_time > self.flags.time:
                buf_stop.set()

        try:
            # Decode Surface
            surf_src = self.dec.decode()
            if surf_src is None:
                self._signal_push_end()
                return False

            # Process to match NN expectations
            surf_inf, surf_osd = self.conv.convert(surf_src)
            if surf_inf is None or surf_osd is None:
                self._signal_push_end()
                return False

            # Run inference
            future = self.executor.submit(
                self._inference, surf_inf, surf_osd, str(self.num_surf))
            self.tasks.add(future)
            future.add_done_callback(self.tasks.remove)

            # Increase surfaces counter
            self.num_surf += 1

        except Exception as e:
            LOGGER.error(
                f"Frame {self.num_surf}. Unexpected excepton: {str(e)}")
            self._signal_push_end()
            return False

        return True

    def all_done(self) -> bool:
        """
        Return True if all frames have been processed
        """
        if not self.push_done:
            return False

        return 0 == len(self.tasks)
