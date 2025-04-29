from typing import List, Optional, Tuple, Callable

Handler = Callable[[List[dict], List[int], List[int]], Tuple[Optional[int], Optional[int]]]
START_INDEX = 0
END_INDEX = -1