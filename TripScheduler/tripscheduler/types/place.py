from typing import TypedDict, List, Literal

class Place(TypedDict):
    id: int
    name: str
    x_cord: float
    y_cord: float
    category: Literal["transport", "restaurant", "accommodation", "landmark"]
    open_time: str        
    close_time: str
    service_time: int
    tags: List[str]
    휴무일: List[str]
    break_time: List[str]
    is_mandatory: bool

PlaceList = List[Place]
