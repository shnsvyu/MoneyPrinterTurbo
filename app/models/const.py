PUNCTUATIONS = [
    "?",
    ",",
    ".",
    "、",
    ";",
    ":",
    "!",
    "…",
    "？",
    "，",
    "。",
    "、",
    "；",
    "：",
    "！",
    "...",
    # 阿拉伯语常用标点也应作为自然断句点，避免脚本文本和 edge-tts
    # 返回的字幕停顿边界不一致，导致后续逐行匹配失败。
    "،",
    "؛",
    "؟",
]

TASK_STATE_FAILED = -1
TASK_STATE_COMPLETE = 1
TASK_STATE_PROCESSING = 4

CROSS_POST_STATE_PENDING = "pending"
CROSS_POST_STATE_PROCESSING = "processing"
CROSS_POST_STATE_COMPLETE = "complete"
CROSS_POST_STATE_FAILED = "failed"

FILE_TYPE_VIDEOS = ["mp4", "mov", "mkv", "webm"]
FILE_TYPE_IMAGES = ["jpg", "jpeg", "png", "bmp"]

# 视频素材来源。locales = 本地文件 + 分片理解入库 ES，供混合检索选片。
VIDEO_SOURCE_PEXELS = "pexels"
VIDEO_SOURCE_PIXABAY = "pixabay"
VIDEO_SOURCE_COVERR = "coverr"
VIDEO_SOURCE_LOCAL = "local"
VIDEO_SOURCE_LOCALES = "locales"
VIDEO_SOURCES = (
    VIDEO_SOURCE_PEXELS,
    VIDEO_SOURCE_PIXABAY,
    VIDEO_SOURCE_COVERR,
    VIDEO_SOURCE_LOCAL,
    VIDEO_SOURCE_LOCALES,
)
LOCAL_VIDEO_SOURCES = (VIDEO_SOURCE_LOCAL, VIDEO_SOURCE_LOCALES)


def normalize_video_source(source: str | None) -> str:
    return str(source or "").strip().lower()


def is_local_video_source(source: str | None) -> bool:
    """本地文件类来源（含普通 local 与 ES 索引 locales）。"""
    return normalize_video_source(source) in LOCAL_VIDEO_SOURCES


def is_locales_video_source(source: str | None) -> bool:
    """本地文件分片理解并写入 Elasticsearch 的来源。"""
    return normalize_video_source(source) == VIDEO_SOURCE_LOCALES


def is_supported_video_source(source: str | None) -> bool:
    return normalize_video_source(source) in VIDEO_SOURCES


# locales 项目分类：索引与检索时写入/过滤 ES 的 project 字段。
LOCALES_PROJECT_INTERNATIONAL_NEWS = "国际新闻"
LOCALES_PROJECT_MOVIE = "电影"
LOCALES_PROJECT_XIANNI = "仙逆"
LOCALES_PROJECT_FANREN = "凡人修仙传"
LOCALES_PROJECTS = (
    LOCALES_PROJECT_INTERNATIONAL_NEWS,
    LOCALES_PROJECT_MOVIE,
    LOCALES_PROJECT_XIANNI,
    LOCALES_PROJECT_FANREN,
)
DEFAULT_LOCALES_PROJECT = LOCALES_PROJECT_INTERNATIONAL_NEWS


def normalize_locales_project(project: str | None) -> str:
    """归一化项目名；无法识别时回退到默认项目。"""
    value = str(project or "").strip()
    # 兼容常见笔误
    aliases = {
        "凡人修仙转": LOCALES_PROJECT_FANREN,
        "fanren": LOCALES_PROJECT_FANREN,
        "xianni": LOCALES_PROJECT_XIANNI,
        "movie": LOCALES_PROJECT_MOVIE,
        "movies": LOCALES_PROJECT_MOVIE,
        "news": LOCALES_PROJECT_INTERNATIONAL_NEWS,
        "international_news": LOCALES_PROJECT_INTERNATIONAL_NEWS,
    }
    if value in aliases:
        return aliases[value]
    if value in LOCALES_PROJECTS:
        return value
    return DEFAULT_LOCALES_PROJECT


def is_supported_locales_project(project: str | None) -> bool:
    value = str(project or "").strip()
    return value in LOCALES_PROJECTS or value in {
        "凡人修仙转",
        "fanren",
        "xianni",
        "movie",
        "movies",
        "news",
        "international_news",
    }

