import python_vali as vali
import cvcuda
import numpy as np
from argparse import Namespace
import logging

LOGGER = logging.getLogger(__file__)


class Composer():
    def __init__(
            self,
            flags: Namespace):

        self.gpu_id = flags.gpu_id
        self.flags = flags

        self.nv_jpeg = vali.PyNvJpegEncoder(self.gpu_id)

        self.jpeg_ctx = self.nv_jpeg.Context(
            compression=100,
            pixel_format=vali.PixelFormat.RGB)

    @staticmethod
    def _transform_bbox(
            bbox: list[float],
            surf_inf: vali.Surface,
            surf_osd: vali.Surface) -> list[np.int32]:
        """
        Transform bbox coordinates
        """
        scale_x = surf_osd.Width / surf_inf.Width
        scale_y = surf_osd.Height / surf_inf.Height

        bbox[0] = int(bbox[0] * scale_x)
        bbox[2] = int(bbox[2] * scale_x)
        bbox[1] = int(bbox[1] * scale_y)
        bbox[3] = int(bbox[3] * scale_y)

        return bbox

    def compress_jpeg(
            self,
            surf_osd: vali.Surface) -> np.ndarray:
        """
        Compress surface into jpeg
        """
        buffers, info = self.nv_jpeg.Run(self.jpeg_ctx, [surf_osd])
        if not len(buffers):
            LOGGER.error(f"Failed to compress jpeg: {str(info)}")
            return None

        return buffers[0]

    def draw(
            self,
            surf_inf: vali.Surface,
            surf_osd: vali.Surface,
            req_body: dict) -> None:
        """
        Do OSD for YOLO output into surf_osd
        """
        nvcv_tensor = cvcuda.as_tensor(surf_osd, "HWC")

        # Process labels and bboxes
        bboxes = []
        labels = []

        for result in req_body['detections']:
            labels.append(result['name'])
            box = result['box']
            bbox = [
                box['x1'],
                box['y1'],
                box['x2'] - box['x1'],
                box['y2'] - box['y1']]

            bboxes.append(Composer._transform_bbox(bbox, surf_inf, surf_osd))

        bbox_list = []
        label_list = []

        for i in range(0, len(bboxes)):
            bbox_list.append(
                cvcuda.BndBoxI(
                    box=tuple(bboxes[i]),
                    thickness=5,
                    borderColor=(0, 255, 0, 255),
                    fillColor=(0, 0, 255, 0)))

            label_list.append(
                cvcuda.Label(
                    utf8Text=labels[i],
                    fontSize=12,
                    tlPos=(bboxes[i][0], bboxes[i][1]),
                    fontColor=(0, 255, 0, 255),
                    bgColor=(0, 0, 255, 0)))

        # Run OSD
        if len(bbox_list) and len(label_list):
            batch_bounding_boxes = cvcuda.Elements(elements=[bbox_list])
            batch_labels = cvcuda.Elements(elements=[label_list])

            cvcuda.osd_into(nvcv_tensor, nvcv_tensor, batch_bounding_boxes)
            cvcuda.osd_into(nvcv_tensor, nvcv_tensor, batch_labels)
