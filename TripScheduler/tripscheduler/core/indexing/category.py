from typing import List

def get_indices_by_category(places: List[dict], category: str) -> List[int]:
    """ 카테고리에 해당하는 장소의 인덱스 반환 """
    return [i for i, p in enumerate(places) if p.get("category") == category]

def validate_place_category(place: dict, expected_category: str, err_msg: str) -> None:
    """ 카테고리 적합 검사 """
    if place.get("category") != expected_category:
        raise ValueError(err_msg)
