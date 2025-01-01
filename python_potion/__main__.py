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
import torchvision
import argparse
import logging
import time
import potion.buffering as buffering
import potion.inference as inference

logger = logging.getLogger(__file__)

if __name__ == "__main__":
    mp.set_start_method("spawn")

    logger = logging.getLogger(__name__)
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

    args = parser.parse_args()

    # Basic agsparse validation
    if args.gpu_id < 0:
        raise RuntimeError("Invalid gpu id, must be >= 0")

    # 1.1
    # Prepare video track params and variable size queue.
    buf_class = buffering.StreamBuffer(
        args.input, {'num_retries': args.num_retries})
    params = buf_class.get_params()
    buf_queue = Queue(maxsize=0)

    # 1.2
    # This process reads video and puts 4kB chunks into variable size queue.
    # It ensures that no frame will be lost if processing is slow.
    buf_proc_stop = mp.Event()
    buf_proc = Process(
        target=buf_class.buf_stream,
        args=(buf_queue, buf_proc_stop),
    )
    buf_proc.start()

    # 2.1
    # Init model here because it takes a long time.
    # Meanwhile buf_stream process will read the video so no frames will be lost
    # during the model initialization.
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    model.to("cuda")

    # 3.1
    # Start inference process. It will take input from queue.
    inf_proc_stop = mp.Event()
    inf_proc = Process(
        target=inference.inference,
        args=(
            buf_queue,
            model,
            inf_proc_stop,
            args.output,
            args.dump,
            buf_class.format_by_codec(),
            int(args.gpu_id),
        ),
    )
    inf_proc.start()

    # Let the script do the job.
    time.sleep(float(args.time))

    # 4.1
    # Stop buf_stream process. No more chunks will be put into variable size queue.
    buf_proc_stop.set()
    buf_proc.join()

    # 4.2
    # Wait for all chunks to be read from variable size queue.
    # Then close it to prevent decoder from reading chunks in endless loop.
    while buf_queue.qsize():
        print(f"Buffer size: {buf_queue.qsize()} chunks left")
        time.sleep(0.1)

    # 5.1
    # Stop inference process.
    inf_proc_stop.set()
    inf_proc.join()
