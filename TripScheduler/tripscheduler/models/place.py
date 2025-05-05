from pydantic import BaseModel
from typing import List, Literal
from pydantic import BaseModel, Field

class PlaceModel(BaseModel):
    id: int
    name: str
    x_cord: float
    y_cord: float
    category: Literal["transport","restaurant","accommodation","landmark"]
    open_time: str
    close_time: str
    service_time: int
    tags: List[str] = []
    closed_days: List[str] = Field(..., alias="휴무일")
    break_times: List[str] = Field(..., alias="break_time")
    is_mandatory: bool = True

    class Config:
        allow_population_by_field_name = True  # JSON → 모델 변환 시 alias 사용 허용
