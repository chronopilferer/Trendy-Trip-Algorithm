# 이미지 밝기 판단 임계값
DARKNESS_THRESHOLD: int      = 30    # 매우 어두운 이미지 제거 
BRIGHTNESS_THRESHOLD: int    = 240   # 너무 밝아 정보가 날아가는 이미지 제거 

# 최소 해상도 
MIN_WIDTH: int               = 256
MIN_HEIGHT: int              = 256

# 텍스트 엔트로피
ENTROPY_THRESHOLD: float     = 3.5    # 너무 단순한 이미지 제거 

# 시각화 기본값 
DEFAULT_BOX_COLOR: tuple     = (0, 255, 0)
DEFAULT_TEXT_COLOR: tuple    = (255, 0, 0)
DEFAULT_LINE_THICKNESS: int  = 2

# 이미지 영역 임계값 
DEFAULT_TEXT_AREA_THRESHOLD: float   = 0.05  # 텍스트가 많은 썸네일류 제거
DEFAULT_PERSON_AREA_THRESHOLD: float = 0.10  # 인물 중심 이미지 제거
DEFAULT_FOOD_AREA_THRESHOLD: float   = 0.6   # 음식 클로즈업 중심 제거

# 이미지 확장자
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# YOLO 객체 타겟
TARGET_LABELS = {
    "person", "bowl", "cup", "sandwich",
    "cake", "bottle", "hot dog", "donut"
}

# CLIP 모델 임계값 
SCENE_THRESHOLD: float  = 0.20   
OBJECT_THRESHOLD: float = 0.30   

# 경계 유예 
MARGIN_DELTA: float = 0.05