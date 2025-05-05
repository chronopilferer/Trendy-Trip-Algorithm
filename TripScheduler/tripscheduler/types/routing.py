from typing import Callable, List, Tuple, Optional
from tripscheduler.types.place import PlaceList

Handler = Callable[
    [PlaceList, List[int], List[int], int, int],
    Tuple[Optional[int], Optional[int]]
]
