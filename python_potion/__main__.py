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

import logging

from multiprocessing import Queue, Process
import multiprocessing as mp
import argparse

import python_potion.common as common
import python_potion.buffering as buffering
import python_potion.client as image_client
import time

LOGGER = logging.getLogger(__file__)


def main(flags: argparse.Namespace) -> None:
    # 1.1
    # Queue with video track chunks has variable size.
    # It serves as temporary storage to prevent data loss if consumer is slow.
    buf_class = buffering.StreamBuffer(flags)
    buf_queue = Queue(maxsize=0)

    # 1.2
    # This process reads video track and puts chunks into variable size queue.
    buf_stop = mp.Event()
    buf_proc = Process(
        target=buf_class.bufferize,
        args=(buf_queue, buf_stop),
    )
    buf_proc.start()

    # 1.3
    # Start wallclock time
    start_time = time.time()

    try:
        # Emergency stop flag.
        all_good = True

        # 2.1
        # Start inference in current process.
        # It will take input from queue, decode and send images to triton inference server.
        client = image_client.ImageClient(flags, buf_queue)

        # 2.2
        # Send inference requests
        # Client will signal buf_proc to stop after timeout
        while client.send_request(buf_stop, start_time):
            pass

        # 2.3
        # Get inference repsonse
        # Once None is received, it means processing is done
        done = False
        while not done:
            res = client.get_response()
            if not res:
                done = True
                break

            print(res)

    except Exception as e:
        all_good = False
        buf_stop.set()
        LOGGER.fatal(str(e))

    finally:
        # 3.1
        # Stop buf_stream process.
        if not all_good:
            common.drain(buf_queue)
        buf_proc.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    try:
        main(common.get_parser().parse_args())
    except Exception as e:
        LOGGER.fatal(str(e))
