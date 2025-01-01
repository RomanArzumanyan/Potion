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

from multiprocessing import Queue
import multiprocessing as mp
import numpy as np
import torch
import torchvision
import logging
from multiprocessing.synchronize import Event as SyncEvent
import potion.decoder as decoder

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


def append_detection(res_json: dict, idx: int, outputs) -> None:
    """
    Append detection to dictionary.

    Args:
        res_json (dict): dict with detections
        idx (int): video frame index
        outputs (_type_): model outputs
    """

    pred_classes = [coco_names[i] for i in outputs[0]["labels"].cpu().numpy()]
    pred_scores = outputs[0]["scores"].detach().cpu().numpy()
    pred_bboxes = outputs[0]["boxes"].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)

    key = "frame " + str(idx)
    frame = {key: []}

    for i, box in enumerate(boxes):
        bbox = {"bbox": box.tolist()}
        detection = {pred_classes[i]: bbox}
        frame[key].append(detection)

    res_json["detections"].append(frame)


def inference(
    inp_queue: Queue,
    model,
    stop_event: SyncEvent,
    output: str,
    dump_fname: str,
    dump_ext: str,
    gpu_id=0,
) -> None:
    """
    Run inference on video frames.
    This function is to be run in separate process.

    Args:
        inp_queue (Queue): video frames queue
        model: torch inference model
        stop_event (SyncEvent): multiprocessing event which stops this funciton
        output(str): path to JSON with detections
        dump_fname (str): dump file name, if empty no dump will be done
        dump_ext (str): dump file dump_ext
        gpu_id (int, optional): GPU to run on. Defaults to 0.
    """

    try:
        dec = decoder.NvDecoder(inp_queue, stop_event,
                                dump_fname, dump_ext, gpu_id)
    except Exception as e:
        logger.fatal(f"Failed to create decoder: {e}")
        return

    idx = 0
    res_json = {"detections": []}

    while not stop_event.is_set():
        try:
            surf = dec.decode()

            # Export to tensor
            img_tensor = torch.from_dlpack(surf)
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
                append_detection(res_json, idx, outputs)
                idx += 1

        except Exception as e:
            logger.error(f"Unexpected exception: {str(e)}")
            break

    with open(output, "w") as f_out:
        f_out.write(json.dumps(res_json, indent=4))
