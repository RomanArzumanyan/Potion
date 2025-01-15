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

from queue import Empty
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

        self.sent_cnt = 0
        self.recv_cnt = 0

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
            "dst_fmt": vali.PixelFormat.RGB_32F_PLANAR,
            "src_w": self.dec.width(),
            "src_h": self.dec.height(),
            "dst_w": self.w,
            "dst_h": self.h
        }

        self.conv = converter.Converter(params, self.flags)

        # Deal with batch size etc.
        self.batch_size = self.flags.batch_size
        self.supports_batching = self.max_batch_size > 0
        if not self.supports_batching and self.batch_size != 1:
            raise RuntimeError("ERROR: This model doesn't support batching.")

        # Async stuff
        self.tasks = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _parse_model(self) -> None:
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(self.model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(self.model_metadata.inputs)))

        if len(self.model_metadata.outputs) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(
                    len(self.model_metadata.outputs))
            )

        if len(self.model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(self.model_config.input)
                )
            )

        input_metadata = self.model_metadata.inputs[0]
        input_config = self.model_config.input[0]
        output_metadata = self.model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception(
                "expecting output datatype to be FP32, model '"
                + self.model_metadata.name
                + "' output type is "
                + output_metadata.datatype
            )

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = self.model_config.max_batch_size > 0
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
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

        if (input_config.format != mc.ModelInput.FORMAT_NCHW) and (
            input_config.format != mc.ModelInput.FORMAT_NHWC
        ):
            raise Exception(
                "unexpected input format "
                + mc.ModelInput.Format.Name(input_config.format)
                + ", expecting "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
                + " or "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
            )

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            self.h = input_metadata.shape[1 if input_batch_dim else 0]
            self.w = input_metadata.shape[2 if input_batch_dim else 1]
            self.c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            self.c = input_metadata.shape[1 if input_batch_dim else 0]
            self.h = input_metadata.shape[2 if input_batch_dim else 1]
            self.w = input_metadata.shape[3 if input_batch_dim else 2]

        self.max_batch_size = self.model_config.max_batch_size
        self.output_name = output_metadata.name
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

        outputs = [grpcclient.InferRequestedOutput(
            self.output_name, class_count=self.flags.classes)]

        return (inputs, outputs)

    @nvtx.annotate()
    def _process(self, results):
        """
        Process inference result and put it into stdout.

        Args:
            results (_type_): Inference result returned by Triton sever

        Raises:
            Exception: if batching is on and result rize doesn't match batch size
        """

        output_array = results.as_numpy(self.output_name)
        assert len(output_array) == self.batch_size

        for results in output_array:
            if not self.supports_batching:
                results = [results]
            for result in results:
                if output_array.dtype.type == np.object_:
                    cls = "".join(chr(x) for x in result).split(":")
                else:
                    cls = result.split(":")
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

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
                str(self.sent_cnt)
            )

            self._process(response)
            self.sent_cnt += 1

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
                return False

            # Process to match NN expectations
            surf_dst = self.conv.convert(surf_src)
            if surf_dst is None:
                return False

            # Download to RAM
            img = np.ndarray(shape=(self.c, self.h, self.w),
                             dtype=triton_to_np_dtype(self.dtype))
            success, info = self.dwn.Run(surf_dst, img)
            if not success:
                LOGGER.error(f"Failed to download surface: {info}")
                return False

            # Create inference request task
            future = self.executor.submit(self._send, [img])
            self.tasks.add(future)
            future.add_done_callback(self.tasks.remove)

        except Exception as e:
            LOGGER.error(
                f"Frame {self.sent_cnt}. Unexpected excepton: {str(e)}")
            return False

        return True

    def complete_requests(self) -> None:
        """
        Won't return unless task set is empty.
        """

        while len(self.tasks):
            time.sleep(0.001)
