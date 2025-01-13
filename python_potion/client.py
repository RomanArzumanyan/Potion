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
import tritonclient.http as httpclient
import python_potion.decoder as decoder
import python_potion.converter as converter
import python_vali as vali
import time
import asyncio

from tritonclient.utils import InferenceServerException
from argparse import Namespace


LOGGER = logging.getLogger(__file__)


class ImageClient():
    def __init__(self, flags: Namespace,):
        """
        Constructor.

        Args:
            flags (Namespace): parsed CLI args

        Raises:
            InferenceServerException: if triton throws an exception
        """

        self.gpu_id = flags.gpu_id

        self.flags = flags

        self.sent_cnt = 0
        self.recv_cnt = 0

        self.res = vali.PySurfaceResizer(
            vali.PixelFormat.RGB_32F, flags.gpu_id)
        self.dwn = vali.PySurfaceDownloader(flags.gpu_id)

        self.triton_client = grpcclient.InferenceServerClient(
            url=self.flags.url, verbose=self.flags.verbose
        )

        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=self.flags.model_name, model_version=self.flags.model_version
            )
        except InferenceServerException as e:
            LOGGER.fatal("failed to retrieve the metadata: " + str(e))
            raise e

        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=self.flags.model_name, model_version=self.flags.model_version
            ).config
        except InferenceServerException as e:
            LOGGER.fatal("failed to retrieve the config: " + str(e))
            raise e

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = self._parse_model(
            self.model_metadata, self.model_config
        )

        self.supports_batching = self.max_batch_size > 0
        if not self.supports_batching and self.flags.batch_size != 1:
            LOGGER.fatal("ERROR: This model doesn't support batching.")
            raise e

        self.batch_size = self.flags.batch_size
        self.sent_cnt = 0
        self.recv_cnt = 0

    async def _send(self, img: list[np.ndarray]) -> None:
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

    def run_loop(self, inp_queue: Queue, buf_stop: SyncEvent,) -> None:
        """
        Inference loop.
        Will set up sync event after run time has passed.

        Args:
            inp_queue (Queue): queue with video track chunks
            buf_stop (SyncEvent): sync event to set up
        """

        try:
            dec = decoder.NvDecoder(inp_queue, self.gpu_id)

        except Exception as e:
            LOGGER.fatal(f"Failed to create decoder: {e}")
            return

        # Asyncio loop and send tasks
        loop = asyncio.get_event_loop()
        tasks = []

        # Lazy init will be done
        conv = None
        surf_dst = vali.Surface.Make(vali.PixelFormat.RGB_32F_PLANAR,
                                     self.w, self.h, gpu_id=0)

        runtime = float(self.flags.time)
        start = time.time()
        while True:
            # Signal stop
            if time.time() - start > runtime:
                buf_stop.set()

            try:
                # Decode Surface
                surf_src = dec.decode()
                if surf_src is None:
                    break

                # Process to match NN expectations
                if not conv:
                    params = {
                        "src_fmt": surf_src.Format,
                        "dst_fmt": surf_dst.Format,
                        "src_w": surf_src.Width,
                        "src_h": surf_src.Height,
                        "dst_w": surf_dst.Width,
                        "dst_h": surf_dst.Height
                    }
                    conv = converter.Converter(params, self.gpu_id)

                conv.run(surf_src, surf_dst)

                # Download to numpy array
                img = np.ndarray(
                    shape=(self.c, self.h, self.w), dtype=np.float32)
                success, info = self.dwn.Run(surf_dst, img)
                if not success:
                    LOGGER.error(f"Failed to download surface: {info}")
                    continue

                # Async send for inference
                tasks.append(loop.create_task(self._send([img.copy()])))

            except Exception as e:
                LOGGER.error(
                    f"Frame {self.sent_cnt}. Unexpected excepton: {str(e)}")
                break

        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    def _parse_model(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(
                    len(model_metadata.outputs))
            )

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)
                )
            )

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception(
                "expecting output datatype to be FP32, model '"
                + model_metadata.name
                + "' output type is "
                + output_metadata.datatype
            )

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = model_config.max_batch_size > 0
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
        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(
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
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        return (
            model_config.max_batch_size,
            input_metadata.name,
            output_metadata.name,
            c,
            h,
            w,
            input_config.format,
            input_metadata.datatype,
        )

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
