import cv2
import numpy as np
from typing import List, Tuple, Union

from utils.constants import DEFAULT_BOX_COLOR, DEFAULT_TEXT_COLOR, DEFAULT_LINE_THICKNESS

def draw_boxes(
    image: np.ndarray,
    boxes: List[Union[Tuple[int,int,int,int], List[Tuple[float,float]]]],
    labels: List[str] = None,
    box_color: Tuple[int,int,int] = DEFAULT_BOX_COLOR,
    text_color: Tuple[int,int,int] = DEFAULT_TEXT_COLOR,
    thickness: int = DEFAULT_LINE_THICKNESS
) -> np.ndarray:
    vis = image.copy().astype(np.uint8)
    for idx, box in enumerate(boxes):
        if isinstance(box, tuple) and len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)
            if labels:
                cv2.putText(vis, labels[idx], (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        else:
            pts = np.array([[int(x), int(y)] for x, y in box], np.int32).reshape(-1,1,2)
            cv2.polylines(vis, [pts], isClosed=True, color=text_color, thickness=thickness)
    return vis
