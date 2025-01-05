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

from multiprocessing import Queue, Process
import multiprocessing as mp
import argparse
import logging
import time
import python_potion.buffering as buffering
import python_potion.client as image_client

LOGGER = None
FLAGS = None

if __name__ == "__main__":
    mp.set_start_method("spawn")

    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        "This sample decodes input video from network and saves is as H.264 video track."
    )

    parser.add_argument(
        "-gpu_id",
        "--gpu-id",
        type=int,
        required=True,
        help="GPU id, check nvidia-smi",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Encoded video file (read from)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output json file name",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        required=True,
        help="processing time, s.",
    )
    parser.add_argument(
        "-d",
        "--dump",
        type=str,
        required=False,
        default="",
        help="dump video filename without extension",
    )
    parser.add_argument(
        "-n",
        "--num_retries",
        type=int,
        required=False,
        default=3,
        help="number of attepts to respawn video reader in case of failure",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-a",
        "--async",
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help="Use asynchronous inference API",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Use streaming inference API. "
        + "The flag is only available with gRPC protocol.",
    )
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=1,
        help="Number of class results to report. Default is 1.",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=str,
        choices=["NONE", "INCEPTION", "VGG"],
        required=False,
        default="NONE",
        help="Type of scaling to apply to image pixels. Default is NONE.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=str,
        required=False,
        default="gRPC",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is gRPC.",
    )

    FLAGS = parser.parse_args()

    # Basic agsparse validation
    if FLAGS.gpu_id < 0:
        raise RuntimeError("Invalid gpu id, must be >= 0")

    # 1.1
    # Queue with video track chunks has variable size.
    # It serves as temporary storage to prevent data loss if consumer is slow.
    buf_class = buffering.StreamBuffer(
        FLAGS.input, {'num_retries': FLAGS.num_retries})
    buf_queue = Queue(maxsize=0)

    # 1.2
    # This process reads video track and puts chunks into variable size queue.
    buf_proc_stop = mp.Event()
    buf_proc = Process(
        target=buf_class.buf_stream,
        args=(buf_queue, buf_proc_stop),
    )
    buf_proc.start()

    # 2.1
    # Start inference process. It will take input from queue, decode and send
    # images to triton inference server.
    inf_class = image_client.ImageClient(FLAGS,
                                         buf_class.format_by_codec())

    inf_proc_stop = mp.Event()
    inf_class.inference_client(buf_queue, inf_proc_stop)
    # inf_proc = Process(
    #     target=inf_class.inference_client,
    #     args=(buf_queue, inf_proc_stop),
    # )
    # inf_proc.start()

    # Let the script do the job.
    time.sleep(float(FLAGS.time))

    # 3.1
    # Stop buf_stream process. No more chunks will be put into queue.
    buf_proc_stop.set()
    buf_proc.join()

    # 3.2
    # Wait for all chunks to be read from variable size queue.
    # Then close it to prevent reading chunks in endless loop.
    while buf_queue.qsize():
        print(f"Buffer size: {buf_queue.qsize()} chunks left")
        time.sleep(0.1)

    # 4.1
    # Stop inference process.
    inf_proc_stop.set()
    inf_proc.join()
