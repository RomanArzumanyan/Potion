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
import numpy as np
import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event as SyncEvent

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import python_potion.decoder as decoder
import python_potion.converter as converter
import python_vali as vali
import time
import concurrent.futures

from tritonclient.utils import InferenceServerException, triton_to_np_dtype
from argparse import Namespace
from enum import Enum

import nvtx


LOGGER = logging.getLogger(__file__)


class ClientState(Enum):
    RUNNING = 0,
    EOF = 1,
    ERROR = 2


class ImageClient():
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

        # Create triton client, parse model metadata and config
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.flags.url, verbose=self.flags.verbose
        )

        self.model_metadata = self.triton_client.get_model_metadata(
            model_name=self.flags.model_name, model_version=self.flags.model_version
        )

        self.model_config = self.triton_client.get_model_config(
            model_name=self.flags.model_name, model_version=self.flags.model_version
        ).config

        self._parse_model()

        # Create decoder, converter, downloader
        self.dwn = vali.PySurfaceDownloader(flags.gpu_id)
        self.dec = decoder.Decoder(inp_queue, self.flags)

        params = {
            "src_fmt": self.dec.format(),
            "dst_fmt": self._get_pix_fmt(),
            "src_w": self.dec.width(),
            "src_h": self.dec.height(),
            "dst_w": self.w,
            "dst_h": self.h
        }

        self.conv = converter.Converter(
            params, self.flags, self.dec.cuda_stream())

        # Deal with batch size etc.
        self.batch_size = self.flags.batch_size
        self.supports_batching = self.max_batch_size > 0
        if not self.supports_batching and self.batch_size != 1:
            raise RuntimeError("ERROR: This model doesn't support batching.")

        # Async stuff
        self.tasks = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Results
        self.results = SimpleQueue()
        self.all_done = False

    def _get_shape(self) -> tuple:
        """
        Get target tensor shape

        Raises:
            RuntimeError: if format isn't supported

        Returns:
            tuple: tuple with shape
        """
        if self.format == mc.ModelInput.FORMAT_NCHW:
            return (self.c, self.h, self.w)
        elif self.format == mc.ModelInput.FORMAT_NHWC:
            return (self.h, self.w, self.c)
        else:
            raise RuntimeError(f"Unsupported shape")

    def _get_pix_fmt(self) -> vali.PixelFormat:
        """
        Get target pixel format from model metadata

        Raises:
            ValueError: if data type or format is not supported

        Returns:
            vali.PixelFormat: target pixel format
        """
        supp_types = ["UINT8", "FP32"]
        if not self.dtype in supp_types:
            raise ValueError(
                f"Unsupported datatype {self.dtype}. Not in {supp_types}")

        supp_fmts = [mc.ModelInput.FORMAT_NHWC, mc.ModelInput.FORMAT_NCHW]
        if not self.format in supp_fmts:
            raise ValueError(
                f"Unsupported format {self.format}. Not in {supp_fmts}")

        if self.dtype == "UINT8":
            return vali.PixelFormat.RGB if self.format == mc.ModelInput.FORMAT_NHWC else vali.PixelFormat.RGB_PLANAR
        else:
            return vali.PixelFormat.RGB_32F if self.format == mc.ModelInput.FORMAT_NHWC else vali.PixelFormat.RGB_32F_PLANAR

    def _parse_model(self) -> None:
        """
        Parse model metadata
        """
        # Only single input models are supported
        if len(self.model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(self.model_metadata.inputs)))

        if len(self.model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(self.model_config.input)
                )
            )

        input_metadata = self.model_metadata.inputs[0]
        input_config = self.model_config.input[0]

        # Input must be picture-alike object of NHWC or NCHW layout
        input_batch_dim = self.model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, self.model_metadata.name, len(
                        input_metadata.shape)
                )
            )

        if type(input_config.format) == str:
            FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
            input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            self.h = input_metadata.shape[1 if input_batch_dim else 0]
            self.w = input_metadata.shape[2 if input_batch_dim else 1]
            self.c = input_metadata.shape[3 if input_batch_dim else 2]
        elif input_config.format == mc.ModelInput.FORMAT_NCHW:
            self.c = input_metadata.shape[1 if input_batch_dim else 0]
            self.h = input_metadata.shape[2 if input_batch_dim else 1]
            self.w = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            raise Exception(f"Unexpected input format")

        self.output_names = []
        for output_metadata in self.model_metadata.outputs:
            self.output_names.append(output_metadata.name)

        self.max_batch_size = self.model_config.max_batch_size
        self.input_name = input_metadata.name
        self.dtype = input_metadata.datatype
        self.format = input_config.format

    def _make_req_data(self, batched_image_data):
        """
        Prepare inference request data

        Args:
            batched_image_data : numpy ndarray or list of that

        Returns:
            tuple with inference inputs and outputs
        """

        inputs = [grpcclient.InferInput(
            self.input_name, batched_image_data.shape, self.dtype)]

        inputs[0].set_data_from_numpy(batched_image_data)

        outputs = []
        for name in self.output_names:
            outputs.append(grpcclient.InferRequestedOutput(
                name, self.flags.classes))

        return (inputs, outputs)

    @nvtx.annotate()
    def _process(self, results) -> None:
        """
        Process inference result and put it into stdout.

        Args:
            results (_type_): Inference result returned by Triton sever

        """
        res = {}
        for name in self.output_names:
            res[name] = results.as_numpy(name)

        self.results.put(res)

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
    def _send(self, img: list[np.ndarray]) -> None:
        """
        Send inference request, get response and write to stdout

        Args:
            img (list[np.ndarray]): images to send
        """

        assert len(img) == self.batch_size

        data = np.stack(img, axis=0) if self.supports_batching else img[0]
        try:
            inputs, outputs = self._make_req_data(data)

            response = self.triton_client.infer(
                self.flags.model_name,
                inputs,
                self.flags.model_version,
                outputs,
                str(self.num_surf)
            )

            self._process(response)
            self.num_resp += 1

        except InferenceServerException as e:
            LOGGER.error("Failed to send inference request: " + str(e))

    @nvtx.annotate()
    def send_request(self, buf_stop: SyncEvent, start_time: float) -> tuple[bool, bool]:
        """
        Submit single inference request.
        If :arg:`buf_stop` is None, requests will be sent until there are decoded frames.
        Otherwise :arg:`buf_stop` shall not be `None` and :arg:`start_time` shall be positive.
        If :arg:`buf_stop`: is not None but :arg:`start_time` is negative, it will be ignored.
        In that case, request will be submitted until timeout is reached.
        After that :arg:`buf_stop` will be set.

        Args:
            buf_stop (SyncEvent, optional): sync event to set up. Defaults to None (process all frames).
            start_time (float, optional): start time, used to check for timeout. Defaults to None.

        Returns:
            bool: True in case of successful request submission, False otherwise.
        """

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

            # Download to RAM
            # Acts as sync point on previous async GPU operations
            img = np.ndarray(shape=self._get_shape(),
                             dtype=triton_to_np_dtype(self.dtype))
            success, info = self.dwn.Run(surf_dst, img)
            if not success:
                LOGGER.error(f"Failed to download surface: {info}")
                self._signal_end()
                return False

            # Create inference request task
            future = self.executor.submit(self._send, [img])
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

    def get_response(self) -> dict:
        """
        Get single inference response.

        Returns:
            dict: dictionary with binary response data. Once None is returned it
            means all responses are taken from sever. All consequent calls will
            return None as well.
        """
        while not self.all_done:
            try:
                res = self.results.get_nowait()
                if not res:
                    self.all_done = True
                return res
            except Empty:
                continue

        return None
