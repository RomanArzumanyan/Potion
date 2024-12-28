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

import json

from typing import Dict
import python_vali as vali
from queue import Empty
from multiprocessing import Queue, Process, Event
import multiprocessing as mp
import numpy as np
import torch
import torchvision
import logging
import os
from multiprocessing.synchronize import Event as SyncEvent
from io import BytesIO

logger = logging.getLogger(__file__)

coco_names = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


class QueueAdapter:
    def __init__(self, inp_queue: Queue, dump: bool, stop_event: SyncEvent,):
        self.inp_queue = inp_queue
        self.stop_event = stop_event
        self.dump = dump

    def read(self, size: int) -> bytes:
        while not self.stop_event.is_set():
            try:
                chunk = self.inp_queue.get_nowait()
                if self.dump:
                    with open("dump.bin", "ab") as f_out:
                        f_out.write(chunk)
                return chunk

            except Empty:
                continue

            except ValueError:
                logger.info("Queue is closed.")
                return bytes()

            except Exception as e:
                logger.error(f"Unexpected excepton: {str(e)}")


def save_result(res_json, idx, boxes, classes) -> None:
    key = "frame " + str(idx)
    frame = {key: []}
    for i, box in enumerate(boxes):
        bbox = {"bbox": box.tolist()}
        detection = {classes[i]: bbox}
        frame[key].append(detection)

    res_json["detections"].append(frame)


def inference(
    inp_queue: Queue,
    model,
    stop_event: SyncEvent,
    output: str,
    dump: bool,
    gpu_id=0,
) -> None:

    # Create decoder
    adapter = QueueAdapter(inp_queue, dump, stop_event)
    py_dec = vali.PyDecoder(adapter, {}, gpu_id)

    # Allocate surfaces just once
    surfaces = [
        vali.Surface.Make(vali.PixelFormat.NV12,
                          py_dec.Width, py_dec.Height, gpu_id),
        vali.Surface.Make(vali.PixelFormat.RGB, py_dec.Width,
                          py_dec.Height, gpu_id),
        vali.Surface.Make(vali.PixelFormat.RGB_PLANAR,
                          py_dec.Width, py_dec.Height, gpu_id),
    ]

    # Color converters + context
    convs = [
        vali.PySurfaceConverter(
            surfaces[0].Format, surfaces[1].Format, gpu_id),
        vali.PySurfaceConverter(
            surfaces[1].Format, surfaces[2].Format, gpu_id),
    ]

    idx = 0
    res_json = {"detections": []}

    while not stop_event.is_set():
        try:
            # Decoding
            success, info = py_dec.DecodeSingleSurface(surfaces[0])
            if not success:
                logger.error(info)
                continue

            # Color conversion
            for i in range(0, len(convs)):
                success, info = convs[i].Run(surfaces[i], surfaces[i + 1])
                if not success:
                    logger.error(info)
                    break

            # Export to tensor
            img_tensor = torch.from_dlpack(surfaces[2])
            img_tensor = img_tensor.clone().detach()
            img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)

            # Apply transformations
            img_tensor = torch.divide(img_tensor, 255.0)
            data_transforms = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            surface_tensor = data_transforms(img_tensor)
            input_batch = surface_tensor.unsqueeze(0).to("cuda")

            # Run inference.
            with torch.no_grad():
                outputs = model(input_batch)

            # Collect segmentation results.
            pred_classes = [coco_names[i]
                            for i in outputs[0]["labels"].cpu().numpy()]
            pred_scores = outputs[0]["scores"].detach().cpu().numpy()
            pred_bboxes = outputs[0]["boxes"].detach().cpu().numpy()
            boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)

            # Save to json
            idx += 1
            save_result(res_json, idx, boxes, pred_classes)

        except Exception as e:
            logger.error(f"Unexpected exception: {str(e)}")
            break

    with open(output, "w") as f_out:
        f_out.write(json.dumps(res_json, indent=4))
