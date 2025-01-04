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

from multiprocessing import Queue
import multiprocessing as mp
import numpy as np
import logging
from multiprocessing.synchronize import Event as SyncEvent

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient

from functools import partial
from tritonclient.utils import InferenceServerException, triton_to_np_dtype


LOGGER = logging.getLogger(__file__)
FLAGS = None


class UserData:
    def __init__(self):
        self._completed_requests = Queue()


class ImageClient:
    def __init__(self, flags):
        FLAGS = flags
        self.triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )

        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version
            )
        except InferenceServerException as e:
            LOGGER.fatal("failed to retrieve the metadata: " + str(e))
            raise e

        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version
            ).config
        except InferenceServerException as e:
            LOGGER.fatal("failed to retrieve the config: " + str(e))
            raise e

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = self.parse_model(
            self.model_metadata, self.model_config
        )

        supports_batching = self.max_batch_size > 0
        if not supports_batching and FLAGS.batch_size != 1:
            LOGGER.fatal("ERROR: This model doesn't support batching.")
            raise e

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        self.requests = []
        self.responses = []
        self.result_filenames = []
        self.request_ids = []
        self.batch = []
        self.image_idx = 0
        self.last_request = False
        self.user_data = UserData()
        self.batch_size = FLAGS.batch_size
        self.sent_cnt = 0
        self.recv_cnt = 0

        self.triton_client.start_stream(
            partial(self.completion_callback, self.user_data))

    def send(self, img: np.ndarray):
        self.batch.append(img)
        if len(self.batch) == self.batch_size:
            inf_batch = np.stack(self.batch, axis=0)
            try:
                self.sent_cnt += 1
                for inputs, outputs, model_name, model_version in self.requestGenerator(
                    inf_batch, self.input_name, self.output_name, self.dtype, FLAGS
                ):
                    self.triton_client.async_stream_infer(
                        FLAGS.model_name,
                        inputs,
                        request_id=str(self.sent_count),
                        model_version=FLAGS.model_version,
                        outputs=outputs,
                    )
            except InferenceServerException as e:
                LOGGER.error("inference failed: " + str(e))

    def recv(self):
        (results, error) = self.user_data._completed_requests.get()
        self.recv_cnt += 1
        if error is not None:
            LOGGER.error("inference failed: " + str(error))
        print(results)

    def completion_callback(self, user_data, result, error):
        user_data._completed_requests.put((result, error))

    def parse_model(self, model_metadata, model_config):
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

    @classmethod
    def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
        protocol = FLAGS.protocol.lower()

        if protocol == "grpc":
            client = grpcclient
        else:
            client = httpclient

        # Set the input data
        inputs = [client.InferInput(
            input_name, batched_image_data.shape, dtype)]
        inputs[0].set_data_from_numpy(batched_image_data)

        outputs = [client.InferRequestedOutput(
            output_name, class_count=FLAGS.classes)]

        yield inputs, outputs, FLAGS.model_name, FLAGS.model_version
