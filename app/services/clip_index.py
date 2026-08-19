"""
本地视频分片理解服务。

流程：
  1. 使用与成片相同的 ``video.build_subclipped_items`` 切分本地视频
  2. 从每段中间抽帧，调用多模态 LLM 生成 caption / tags
  3. 写入 Elasticsearch，供后续按文案/关键词检索素材分片

该服务完全 opt-in：``clip_index_enabled=true`` 且安装 elasticsearch 后才生效。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import VideoConcatMode
from app.services import clip_es, video
from app.utils import file_security, utils

_CAPTION_PROMPT = """你是视频素材分析助手。根据给定关键帧，用简洁中文描述该视频片段画面内容。
只输出 JSON，不要 Markdown：
{"caption":"一句话画面描述","tags":["标签1","标签2","标签3"],"language":"zh"}
要求：caption 侧重可检索的视觉元素（场景、主体、动作、氛围）；tags 3-8 个短词。"""

# 视觉调用节流：低 RPM 账号可在配置里加大间隔；高并发默认不人为等待。
_DEFAULT_VISION_MIN_INTERVAL = 0.0
_DEFAULT_VISION_MAX_RETRIES = 5
_DEFAULT_VISION_RETRY_BASE = 2.0
_last_vision_call_at = 0.0


class ClipUnderstandError(RuntimeError):
    """分片理解失败。"""


class ClipVisionRateLimitError(ClipUnderstandError):
    """视觉接口触发限流，经过重试后仍失败。"""


def _vision_min_interval() -> float:
    value = clip_es.get_setting(
        "clip_index_vision_min_interval", _DEFAULT_VISION_MIN_INTERVAL
    )
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_VISION_MIN_INTERVAL


def _vision_max_retries() -> int:
    value = clip_es.get_setting(
        "clip_index_vision_max_retries", _DEFAULT_VISION_MAX_RETRIES
    )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return _DEFAULT_VISION_MAX_RETRIES


def _vision_retry_base() -> float:
    value = clip_es.get_setting(
        "clip_index_vision_retry_base_seconds", _DEFAULT_VISION_RETRY_BASE
    )
    try:
        return max(0.5, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_VISION_RETRY_BASE


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "429" in text or "rate_limit" in text or "rate limit" in text:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    return "rate_limit" in str(body).lower()


def _retry_after_seconds(exc: BaseException, attempt: int) -> float:
    """从错误信息解析等待秒数；解析失败则指数退避。"""
    text = str(exc)
    match = re.search(
        r"try again after\s+(\d+(?:\.\d+)?)\s*seconds?", text, re.IGNORECASE
    )
    if match:
        return max(1.0, float(match.group(1)))
    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return max(1.0, float(match.group(1)))
    base = _vision_retry_base()
    return min(60.0, base * (2 ** max(0, attempt)))


def _wait_for_vision_slot() -> None:
    """请求前按最小间隔节流，降低触发组织 RPM 上限的概率。"""
    global _last_vision_call_at
    interval = _vision_min_interval()
    if interval <= 0:
        return
    now = time.monotonic()
    wait = (_last_vision_call_at + interval) - now
    if wait > 0:
        logger.info(f"vision rate limit pacing: sleep {wait:.1f}s before next request")
        time.sleep(wait)


def _mark_vision_call() -> None:
    global _last_vision_call_at
    _last_vision_call_at = time.monotonic()


def is_enabled() -> bool:
    return clip_es.is_enabled()


def _local_videos_dir() -> str:
    return utils.storage_dir("local_videos", create=True)


def resolve_local_video_path(unsafe_path: str) -> str:
    """将用户传入的文件名/路径解析到 local_videos 目录内。"""
    return file_security.resolve_path_within_directory(
        _local_videos_dir(), unsafe_path
    )


def list_local_video_files() -> list[str]:
    root = Path(_local_videos_dir())
    exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(str(path))
    return files


def _file_content_hash(file_path: str) -> str:
    stat = os.stat(file_path)
    raw = f"{file_path}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _filename_meta_path(source_path: str) -> str:
    return f"{source_path}.meta.json"


def write_original_filename_meta(source_path: str, original_filename: str) -> None:
    """在 UUID 落盘文件旁记录上传前的原始文件名。"""
    name = Path(str(original_filename or "").strip()).name
    if not source_path or not name:
        return
    meta_path = _filename_meta_path(source_path)
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"original_filename": name}, fh, ensure_ascii=False)
    except OSError as exc:
        logger.warning(f"write original filename meta failed: {meta_path}: {exc}")


def read_original_filename_meta(source_path: str) -> str:
    meta_path = _filename_meta_path(source_path)
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Path(str(data.get("original_filename") or "").strip()).name
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""


def resolve_display_filename(
    source_path: str, filename: str | None = None
) -> str:
    """优先使用显式传入名 / sidecar 原始名，否则回退到磁盘文件名。"""
    explicit = Path(str(filename or "").strip()).name
    if explicit:
        return explicit
    meta_name = read_original_filename_meta(source_path)
    if meta_name:
        return meta_name
    return Path(source_path).name


def clip_document_id(
    source_path: str,
    start_time: float,
    end_time: float,
    project: str = "",
) -> str:
    raw = f"{project}|{source_path}|{start_time:.3f}|{end_time:.3f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_keyframe_jpeg(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str | None = None,
) -> str:
    """从分片时间窗口中点抽取一帧 JPEG。"""
    mid = max(0.0, (float(start_time) + float(end_time)) / 2.0)
    clip = video._open_video_clip_quietly(video_path)
    try:
        # 避免中点刚好落在尾帧之外。
        t = min(mid, max(0.0, clip.duration - 0.05))
        frame = clip.get_frame(t)
    finally:
        video.close_clip(clip)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ClipUnderstandError(
            "Pillow is required for keyframe extraction"
        ) from exc

    image = Image.fromarray(frame)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".jpg", prefix="mpt-clip-frame-")
        os.close(fd)
    image.convert("RGB").save(output_path, format="JPEG", quality=85)
    return output_path


def _parse_understanding_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ClipUnderstandError(f"invalid understanding response: {text!r}")
        data = json.loads(match.group(0))

    caption = str(data.get("caption") or "").strip()
    if not caption:
        raise ClipUnderstandError("understanding response missing caption")
    tags_raw = data.get("tags") or []
    if isinstance(tags_raw, str):
        tags = [t.strip() for t in re.split(r"[,，]", tags_raw) if t.strip()]
    else:
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    language = str(data.get("language") or "zh").strip() or "zh"
    return {"caption": caption, "tags": tags[:12], "language": language}


def understand_frame_with_llm(image_path: str) -> dict[str, Any]:
    """
    使用当前配置的 OpenAI-compatible / Gemini 多模态能力理解关键帧。

    限流会按配置重试/等待；最终仍失败则抛错，由索引流程跳过该分片，
    **不会**再用弱描述写入 Elasticsearch。
    """
    with open(image_path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("ascii")

    provider_id = str(config.app.get("llm_provider") or "openai").lower()
    max_retries = _vision_max_retries()
    last_error: BaseException | None = None

    for attempt in range(max_retries):
        try:
            _wait_for_vision_slot()
            result = _vision_chat_completion(provider_id, image_b64)
            _mark_vision_call()
            return _parse_understanding_json(result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_rate_limit_error(exc) and attempt + 1 < max_retries:
                delay = _retry_after_seconds(exc, attempt)
                logger.warning(
                    f"vision rate limited (attempt {attempt + 1}/{max_retries}), "
                    f"retry in {delay:.1f}s: {exc}"
                )
                time.sleep(delay)
                _mark_vision_call()
                continue
            if _is_rate_limit_error(exc):
                raise ClipVisionRateLimitError(
                    f"vision rate limited after {max_retries} attempts: {exc}"
                ) from exc
            raise ClipUnderstandError(f"vision understanding failed: {exc}") from exc

    raise ClipUnderstandError(
        f"vision understanding failed: {last_error}"
    )


def _vision_chat_completion(provider_id: str, image_b64: str) -> str:
    from app.models.llm_provider import get_llm_provider

    provider = get_llm_provider(provider_id)
    if provider is None:
        raise ClipUnderstandError(f"unsupported llm provider: {provider_id}")

    api_key = config.app.get(provider.config_key("api_key"), "") or ""
    model_name = provider.resolve_model_name(
        config.app.get(provider.config_key("model_name"), "")
    )
    base_url = provider.resolve_base_url(
        config.app.get(provider.config_key("base_url"), "")
    )

    if provider.adapter == "gemini":
        return _vision_gemini(api_key, model_name, image_b64)

    # OpenAI-compatible 多模态消息（Moonshot / OpenAI / 多数中转站）。
    from openai import OpenAI

    if not api_key and provider.requires_api_key:
        raise ClipUnderstandError(f"{provider_id}: api_key is not set")
    client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None)
    # 部分模型（如部分 Moonshot / Kimi）只允许 temperature=1；
    # 可通过 [es] clip_index_vision_temperature 覆盖。
    temperature = float(
        clip_es.get_setting("clip_index_vision_temperature", 1) or 1
    )
    create_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
    }
    try:
        resp = client.chat.completions.create(**create_kwargs)
    except Exception as exc:  # noqa: BLE001
        # 模型拒绝自定义 temperature 时，强制 1 再试一次。
        message = str(exc).lower()
        if "temperature" in message and temperature != 1 and not _is_rate_limit_error(exc):
            create_kwargs["temperature"] = 1
            resp = client.chat.completions.create(**create_kwargs)
        else:
            raise
    content = resp.choices[0].message.content if resp.choices else ""
    if not content:
        raise ClipUnderstandError(f"{provider_id}: empty vision response")
    return str(content)


def _vision_gemini(api_key: str, model_name: str, image_b64: str) -> str:
    if not api_key:
        raise ClipUnderstandError("gemini: api_key is not set")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    image_bytes = base64.b64decode(image_b64)
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=_CAPTION_PROMPT),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
        ],
    )
    text = getattr(resp, "text", None) or ""
    if not text:
        raise ClipUnderstandError("gemini: empty vision response")
    return str(text)


def build_clip_document(
    item: video.SubClippedVideoClip,
    understanding: dict[str, Any],
    content_hash: str,
    project: str = "",
    filename: str | None = None,
) -> dict[str, Any]:
    from app.models import const

    source_path = item.source_file_path or item.file_path
    start = float(item.start_time or 0.0)
    end = float(item.end_time or start)
    disk_name = Path(source_path).name
    display_name = resolve_display_filename(source_path, filename)
    project_name = const.normalize_locales_project(project)
    tags = list(understanding.get("tags") or [])
    return {
        "clip_id": clip_document_id(source_path, start, end, project=project_name),
        "project": project_name,
        "filename": display_name,
        "source_path": source_path,
        "source_name": disk_name,
        "start_time": start,
        "end_time": end,
        "duration": float(item.duration),
        "width": int(item.width or 0),
        "height": int(item.height or 0),
        "caption": understanding["caption"],
        "tags": tags,
        "language": understanding.get("language") or "zh",
        "content_hash": content_hash,
    }


def index_local_video(
    video_path: str,
    *,
    project: str = "",
    filename: str | None = None,
    max_clip_duration: int | None = None,
    clip_speed: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    """
    对单个本地视频执行：分片 → 理解 → 写入 ES。

    文档必含：project / filename / tags / start_time / end_time。
    filename 为上传前的原始文件名（若有）。
    """
    from app.models import const

    if not is_enabled():
        raise ClipUnderstandError(
            "clip index is disabled; set clip_index_enabled=true in config.toml"
        )

    project_name = const.normalize_locales_project(project)
    resolved = resolve_local_video_path(video_path)
    if not os.path.isfile(resolved):
        raise ClipUnderstandError(f"video not found: {video_path}")

    display_name = resolve_display_filename(resolved, filename)
    if filename and Path(str(filename).strip()).name:
        write_original_filename_meta(resolved, display_name)

    clip_duration = int(
        max_clip_duration
        or clip_es.get_setting("clip_index_segment_seconds")
        or config.app.get("video_clip_duration")
        or 5
    )
    content_hash = _file_content_hash(resolved)

    # force 预留给调用方显式重建语义；当前与默认行为一致。
    _ = force

    items = video.build_subclipped_items(
        video_paths=[resolved],
        max_clip_duration=clip_duration,
        clip_speed=clip_speed,
        video_concat_mode=VideoConcatMode.random,
        # 索引需要保留全部时间片，不做“每源只留最长段”的去重。
        prioritize_unique=False,
    )

    documents: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in items:
        frame_path = None
        try:
            frame_path = extract_keyframe_jpeg(
                item.file_path, item.start_time, item.end_time
            )
            understanding = understand_frame_with_llm(frame_path)
            documents.append(
                build_clip_document(
                    item,
                    understanding,
                    content_hash,
                    project=project_name,
                    filename=display_name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"[{project_name}] {display_name} "
                f"[{item.start_time:.2f}-{item.end_time:.2f}]: {exc}"
            )
            logger.error(f"clip understand failed (not indexed): {msg}")
            errors.append(msg)
        finally:
            if frame_path and os.path.isfile(frame_path):
                try:
                    os.remove(frame_path)
                except OSError:
                    pass

    deleted = 0
    written = 0
    if not documents:
        logger.warning(
            f"skip ES write for {display_name} project={project_name}: "
            "no successful vision captions (avoid storing weak/fallback text)"
        )
    elif len(documents) == len(items):
        deleted = clip_es.delete_by_source(resolved, project=project_name)
        written = clip_es.bulk_upsert_clips(documents)
    else:
        written = clip_es.bulk_upsert_clips(documents)
        logger.warning(
            f"partial index for {display_name} project={project_name}: "
            f"{written}/{len(items)} clips written, "
            f"{len(errors)} failed segments skipped (old docs kept)"
        )

    return {
        "source_path": resolved,
        "source_name": Path(resolved).name,
        "project": project_name,
        "filename": display_name,
        "segments": len(items),
        "indexed": written,
        "deleted_old": deleted,
        "content_hash": content_hash,
        "errors": errors,
    }


def index_all_local_videos(**kwargs) -> dict[str, Any]:
    files = list_local_video_files()
    results = []
    for path in files:
        try:
            results.append(index_local_video(path, **kwargs))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"index_local_video failed for {path}: {exc}")
            results.append(
                {
                    "source_path": path,
                    "project": kwargs.get("project") or "",
                    "indexed": 0,
                    "errors": [str(exc)],
                }
            )
    return {
        "total_files": len(files),
        "project": kwargs.get("project") or "",
        "results": results,
        "indexed": sum(int(r.get("indexed") or 0) for r in results),
    }


def split_script_units(script: str) -> list[str]:
    """把脚本拆成检索用的句子/短段，过滤过短噪声。"""
    text = (script or "").strip()
    if not text:
        return []
    # 先按段落，再按中英文句号/问叹号切开。
    chunks: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = re.split(r"(?<=[。！？.!?])\s*", paragraph)
        for part in parts:
            unit = part.strip(" \t\r\n\"'“”‘’")
            if len(unit) >= 4:
                chunks.append(unit)
    if not chunks and text:
        chunks = [text]
    return chunks


def normalize_terms(video_terms) -> list[str]:
    if not video_terms:
        return []
    if isinstance(video_terms, str):
        terms = [t.strip() for t in re.split(r"[,，]", video_terms)]
    elif isinstance(video_terms, list):
        terms = [str(t).strip() for t in video_terms]
    else:
        return []
    return [t for t in terms if t]


def _normalize_source_path(path: str | None) -> str:
    """统一 Windows 路径大小写/符号链接，避免 allowed 过滤误杀。"""
    value = str(path or "").strip()
    if not value:
        return ""
    try:
        return os.path.normcase(os.path.realpath(value))
    except OSError:
        return os.path.normcase(os.path.abspath(value))


def _rrf_score(rank: int, weight: float, es_score: float | None = None) -> float:
    """Reciprocal Rank Fusion；可选叠加归一化的 ES 分数作为微调。"""
    base = weight / (60.0 + rank + 1)
    if es_score is None:
        return base
    # ES BM25 分数量级不稳定，只做轻量加成。
    return base + weight * 0.01 * float(es_score)


def hybrid_rank_clips(
    video_script: str,
    video_terms=None,
    *,
    project: str = "",
    size_per_query: int = 8,
    sentence_weight: float | None = None,
    term_weight: float | None = None,
    allowed_source_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    混合检索：脚本句子 + 关键词，用 RRF 融合后按「句序优先」排出时间线候选。

    返回的每个元素都带 `_hybrid_score`、`_match_sources`（sentence/term）。
    """
    from app.models import const

    if not is_enabled():
        raise ClipUnderstandError(
            "clip index is disabled; set clip_index_enabled=true in config.toml"
        )

    project_name = const.normalize_locales_project(project) if project else ""
    sentence_weight = float(
        sentence_weight
        if sentence_weight is not None
        else clip_es.get_setting("clip_index_sentence_weight", 0.6)
    )
    term_weight = float(
        term_weight
        if term_weight is not None
        else clip_es.get_setting("clip_index_term_weight", 0.4)
    )
    sentences = split_script_units(video_script)
    terms = normalize_terms(video_terms)
    if not sentences and not terms:
        logger.warning(
            "hybrid search skipped: empty video_script and video_terms "
            f"(project={project_name or '-'})"
        )
        return []

    allowed_norm: set[str] | None = None
    if allowed_source_paths:
        allowed_norm = {
            _normalize_source_path(path)
            for path in allowed_source_paths
            if path
        }
        allowed_norm.discard("")

    # clip_id -> accumulated doc
    merged: dict[str, dict[str, Any]] = {}
    # 每个句子对应的排序 hit 列表，用于后续按句序挑片
    sentence_hits: list[list[dict[str, Any]]] = []

    def _accept(hit: dict[str, Any]) -> bool:
        if project_name and hit.get("project") not in (None, "", project_name):
            # 旧文档可能没有 project 字段；仅当明确写入其它项目时才过滤。
            if hit.get("project"):
                return False
        if not allowed_norm:
            return True
        return _normalize_source_path(hit.get("source_path")) in allowed_norm

    def _accumulate(hit: dict[str, Any], score: float, source: str) -> None:
        clip_id = str(hit.get("clip_id") or "").strip()
        if not clip_id or not _accept(hit):
            return
        existing = merged.get(clip_id)
        if existing is None:
            item = dict(hit)
            item["_hybrid_score"] = score
            item["_match_sources"] = {source}
            merged[clip_id] = item
        else:
            existing["_hybrid_score"] = float(existing.get("_hybrid_score") or 0) + score
            sources = existing.setdefault("_match_sources", set())
            if isinstance(sources, set):
                sources.add(source)
            else:
                existing["_match_sources"] = set(sources) | {source}

    for sentence in sentences:
        try:
            hits = clip_es.search_clips(
                sentence, size=size_per_query, project=project_name or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"sentence search failed: {exc}")
            hits = []
        filtered = [h for h in hits if _accept(h)]
        sentence_hits.append(filtered)
        for rank, hit in enumerate(filtered):
            _accumulate(
                hit,
                _rrf_score(rank, sentence_weight, hit.get("_score")),
                "sentence",
            )

    for term in terms:
        try:
            hits = clip_es.search_clips(
                term, size=size_per_query, project=project_name or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"term search failed: {exc}")
            hits = []
        for rank, hit in enumerate(hits):
            if not _accept(hit):
                continue
            _accumulate(
                hit,
                _rrf_score(rank, term_weight, hit.get("_score")),
                "term",
            )

    # 句序优先：逐句取该句最佳未用片段，再按融合分补齐。
    ordered: list[dict[str, Any]] = []
    used: set[str] = set()
    for hits in sentence_hits:
        for hit in hits:
            clip_id = str(hit.get("clip_id") or "")
            if not clip_id or clip_id in used:
                continue
            doc = merged.get(clip_id) or dict(hit)
            ordered.append(doc)
            used.add(clip_id)
            break

    leftovers = sorted(
        (doc for cid, doc in merged.items() if cid not in used),
        key=lambda d: float(d.get("_hybrid_score") or 0),
        reverse=True,
    )
    ordered.extend(leftovers)

    for doc in ordered:
        sources = doc.get("_match_sources")
        if isinstance(sources, set):
            doc["_match_sources"] = sorted(sources)
    return ordered


def select_clips_for_duration(
    ranked_clips: list[dict[str, Any]],
    *,
    audio_duration: float,
    max_clip_duration: float | None = None,
) -> list[dict[str, Any]]:
    """按旁白时长贪心选取分片，尽量覆盖 audio_duration + 安全余量。"""
    target = max(0.0, float(audio_duration)) + 0.1
    limit = float(max_clip_duration or 0) or None
    selected: list[dict[str, Any]] = []
    total = 0.0
    for clip in ranked_clips:
        if total >= target:
            break
        duration = float(clip.get("duration") or 0)
        if duration <= 0:
            start = float(clip.get("start_time") or 0)
            end = float(clip.get("end_time") or start)
            duration = max(0.0, end - start)
        if duration <= 0:
            continue
        if limit:
            duration = min(duration, limit)
        selected.append(clip)
        total += duration
    return selected


def export_clip_segment(
    source_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> str:
    """把命中的时间窗导出为独立短视频，供 combine_videos 顺序拼接。"""
    clip = video._open_video_clip_quietly(source_path)
    try:
        start = max(0.0, float(start_time))
        end = min(float(end_time), float(clip.duration))
        if end <= start:
            raise ClipUnderstandError(
                f"invalid clip window: {start_time}-{end_time} for {source_path}"
            )
        segment = clip.subclipped(start, end)
        try:
            video._write_videofile_with_codec_fallback(
                segment,
                output_path,
                codec=video._get_configured_video_codec(),
                logger=None,
                fps=video.fps,
            )
        finally:
            video.close_clip(segment)
    finally:
        video.close_clip(clip)
    return output_path


def ensure_sources_indexed(
    source_paths: list[str],
    *,
    project: str = "",
    max_clip_duration: int | None = None,
    filename_by_path: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    确保 locales 用到的本地源文件已写入 ES（按项目维度）。

    已有分片文档的源文件跳过；缺失的自动走 index_local_video。
    """
    from app.models import const

    if not is_enabled():
        raise ClipUnderstandError(
            "locales requires clip index; set [es] clip_index_enabled=true "
            "and configure elasticsearch_hosts"
        )

    project_name = const.normalize_locales_project(project)
    name_map = filename_by_path or {}
    indexed = 0
    skipped = 0
    errors: list[str] = []
    for path in source_paths:
        if not path or not os.path.isfile(path):
            continue
        display_name = resolve_display_filename(path, name_map.get(path))
        try:
            if clip_es.source_has_clips(path, project=project_name):
                skipped += 1
                continue
            logger.info(
                f"locales auto-index missing source: "
                f"project={project_name} file={display_name}"
            )
            result = index_local_video(
                path,
                project=project_name,
                filename=display_name,
                max_clip_duration=max_clip_duration,
                force=True,
            )
            indexed += int(result.get("indexed") or 0)
            errors.extend(result.get("errors") or [])
        except Exception as exc:  # noqa: BLE001
            msg = f"[{project_name}] {display_name}: {exc}"
            logger.error(f"locales auto-index failed: {msg}")
            errors.append(msg)
    return {
        "project": project_name,
        "indexed_segments": indexed,
        "skipped_sources": skipped,
        "errors": errors,
    }


def materialize_hybrid_clips(
    task_id: str,
    video_script: str,
    video_terms=None,
    *,
    project: str = "",
    audio_duration: float,
    max_clip_duration: int | None = None,
) -> list[str]:
    """
    按所选项目混合检索 → 按时长选型 → 导出到任务目录。

    只按 project 过滤 ES；不限制本次上传文件路径。
    导出时使用 ES 中的 filename / start_time / end_time，供后续合并成片。
    """
    from app.models import const

    project_name = const.normalize_locales_project(project)
    ranked = hybrid_rank_clips(
        video_script=video_script,
        video_terms=video_terms,
        project=project_name,
        allowed_source_paths=None,
    )
    if not ranked:
        try:
            project_docs = clip_es.count_by_project(project_name)
        except Exception:  # noqa: BLE001
            project_docs = -1
        logger.warning(
            f"hybrid clip search returned no hits for project={project_name} "
            f"(indexed_docs≈{project_docs}, "
            f"script_units={len(split_script_units(video_script))}, "
            f"terms={len(normalize_terms(video_terms))})"
        )
        return []

    clip_duration = int(
        max_clip_duration
        or clip_es.get_setting("clip_index_segment_seconds")
        or config.app.get("video_clip_duration")
        or 5
    )
    selected = select_clips_for_duration(
        ranked,
        audio_duration=audio_duration,
        max_clip_duration=clip_duration,
    )
    if not selected:
        return []

    output_dir = utils.task_dir(task_id)
    os.makedirs(output_dir, exist_ok=True)
    exported: list[str] = []
    for index, clip in enumerate(selected, start=1):
        source_path = str(clip.get("source_path") or "")
        if not source_path or not os.path.isfile(source_path):
            logger.warning(f"skip missing hybrid source: {source_path}")
            continue
        out_path = os.path.join(output_dir, f"matched-clip-{index:03d}.mp4")
        try:
            export_clip_segment(
                source_path,
                float(clip.get("start_time") or 0),
                float(clip.get("end_time") or 0),
                out_path,
            )
            exported.append(out_path)
            logger.info(
                "hybrid matched clip "
                f"{index}: project={clip.get('project') or project_name} "
                f"file={clip.get('filename') or Path(source_path).name} "
                f"tags={clip.get('tags')} "
                f"[{clip.get('start_time'):.2f}-{clip.get('end_time'):.2f}] "
                f"sources={clip.get('_match_sources')} "
                f"score={float(clip.get('_hybrid_score') or 0):.4f} "
                f"caption={str(clip.get('caption') or '')[:60]}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"export hybrid clip failed: {exc}")

    return exported


def search_local_clips(
    query: str,
    size: int = 10,
    project: str = "",
) -> list[dict[str, Any]]:
    if not is_enabled():
        raise ClipUnderstandError(
            "clip index is disabled; set clip_index_enabled=true in config.toml"
        )
    from app.models import const

    project_name = const.normalize_locales_project(project) if project else None
    return clip_es.search_clips(
        query, size=size, project=project_name or None
    )


def search_local_clips_hybrid(
    query: str = "",
    *,
    video_script: str = "",
    video_terms=None,
    project: str = "",
    size: int = 10,
) -> list[dict[str, Any]]:
    """API 用：既支持单 query，也支持脚本+关键词混合（可按项目过滤）。"""
    script = (video_script or query or "").strip()
    ranked = hybrid_rank_clips(
        video_script=script,
        video_terms=video_terms
        if video_terms is not None
        else ([query] if query and not video_script else None),
        project=project,
        size_per_query=max(size, 5),
    )
    return ranked[: max(1, min(int(size), 100))]
