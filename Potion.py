from multiprocessing import Queue, Process
import multiprocessing as mp
import torchvision
import argparse
import logging
import time
import Buffering
import Inference
import os

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
        type=bool,
        required=False,
        default=False,
        help="dump video track to ./dump.bin",
    )

    args = parser.parse_args()

    # 1.1
    # Prepare video track params and variable size queue.
    params = Buffering.get_stream_params(args.input)
    buf_queue = Queue(maxsize=0)

    # 1.2
    # This process reads video and puts 4kB chunks into variable size queue.
    # It ensures that no frame will be lost if processing is slow.
    buf_proc_stop = mp.Event()
    buf_proc = Process(
        target=Buffering.buf_stream,
        args=(args.input, params, buf_queue, buf_proc_stop)
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
        target=Inference.inference,
        args=(
            buf_queue,
            model,
            inf_proc_stop,
            args.output,
            args.dump,
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
