from typing import TypedDict, Dict

class ImageMetadata(TypedDict, total=False):
    img_path: str
    img_name: str
    category: str

    brightness_score: float
    image_width: float
    image_height: float
    resolution_ratio: float
    entropy_score: float

    text_area_ratio: float
    text_box_count: float

    person_area_ratio: float
    food_area_ratio: float

    scene_max: float
    scene_topk_avg: float
    object_max: float
    object_topk_avg: float
    gap_max: float
    gap_avg: float

    flags: Dict[str, bool]
    is_pass: bool
