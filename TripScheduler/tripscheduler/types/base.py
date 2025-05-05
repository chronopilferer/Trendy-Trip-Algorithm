from typing import TypedDict, Literal

class DayInfo(TypedDict):
    is_first_day: bool
    is_last_day: bool
    date: str             
    weekday: Literal[
        "월요일", "화요일", "수요일", "목요일",
        "금요일", "토요일", "일요일"
    ]
