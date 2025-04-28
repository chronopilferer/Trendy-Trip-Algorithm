# 이미지 밝기 판단 임계값
DARKNESS_THRESHOLD: int      = 40
BRIGHTNESS_THRESHOLD: int    = 220

# 최소 해상도
MIN_WIDTH: int               = 300
MIN_HEIGHT: int              = 300

# 텍스트 엔트로피
ENTROPY_THRESHOLD: float     = 3.5

# 시각화 기본값
DEFAULT_BOX_COLOR: tuple     = (0, 255, 0)
DEFAULT_TEXT_COLOR: tuple    = (255, 0, 0)
DEFAULT_LINE_THICKNESS: int  = 2

# 이미지 영역 임계값 
DEFAULT_TEXT_AREA_THRESHOLD: float = 0.1
DEFAULT_PERSON_AREA_THRESHOLD: float = 0.3
DEFAULT_FOOD_AREA_THRESHOLD: float = 0.5

# 이미지 확장자 
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# YOLO 객체 타겟
TARGET_LABELS = { 'person', 'bowl', 'cup', 'sandwich', 'cake', 'bottle', 'hot dog', 'donut' }

# clip 모델 임계값
SCENE_THRESHOLD = 0.30  
OBJECT_THRESHOLD = 0.30
MARGIN_DELTA = 0.05