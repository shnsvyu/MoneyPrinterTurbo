"""
Elasticsearch 存储层：本地视频分片理解结果的索引与检索。

依赖可选：`uv sync --extra elasticsearch`。未安装或未启用时，调用方应
先检查 ``is_enabled()``，避免把 ES 变成主流程硬依赖。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from app.config import config

DEFAULT_INDEX = "mpt_local_clips"
INDEXED_AT_FORMAT = "%Y-%m-%d %H:%M:%S"
_es_client = None


def format_indexed_at(when: datetime | None = None) -> str:
    """索引时间：yyyy-MM-dd HH:mm:ss（本地时区）。"""
    return (when or datetime.now()).strftime(INDEXED_AT_FORMAT)


class ClipIndexError(RuntimeError):
    """Clip 索引/检索相关错误。"""


def get_setting(key: str, default=None):
    """优先读 [es]，再回退 [app]，兼容两种配置写法。"""
    es_cfg = getattr(config, "es", None) or {}
    if key in es_cfg and es_cfg.get(key) not in (None, ""):
        return es_cfg.get(key)
    return config.app.get(key, default)


def is_enabled() -> bool:
    return bool(get_setting("clip_index_enabled", False))


def _hosts() -> list[str]:
    hosts = get_setting("elasticsearch_hosts") or ["http://127.0.0.1:9200"]
    if isinstance(hosts, str):
        return [hosts.strip()] if hosts.strip() else ["http://127.0.0.1:9200"]
    return [str(h).strip() for h in hosts if str(h).strip()]


def index_name() -> str:
    name = str(get_setting("clip_index_name") or DEFAULT_INDEX).strip()
    return name or DEFAULT_INDEX


def get_client():
    """懒加载 Elasticsearch 客户端；依赖未安装时抛出明确错误。"""
    global _es_client
    if _es_client is not None:
        return _es_client
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ClipIndexError(
            "elasticsearch package is not installed; "
            "run: uv sync --extra elasticsearch"
        ) from exc

    kwargs: dict[str, Any] = {"hosts": _hosts()}
    api_key = str(get_setting("elasticsearch_api_key") or "").strip()
    username = str(get_setting("elasticsearch_username") or "").strip()
    password = str(get_setting("elasticsearch_password") or "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    elif username:
        kwargs["basic_auth"] = (username, password)

    verify = config.app.get("tls_verify", True)
    kwargs["verify_certs"] = bool(verify)
    _es_client = Elasticsearch(**kwargs)
    return _es_client


def reset_client() -> None:
    """测试用：清空缓存的客户端。"""
    global _es_client
    _es_client = None


def _index_properties(dims: int = 0) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "clip_id": {"type": "keyword"},
        # 项目 / 上传前文件名 / tags / 起止时间：后续按这些字段检索并合并成片
        "project": {"type": "keyword"},
        "filename": {"type": "keyword"},
        "source_path": {"type": "keyword"},
        "source_name": {"type": "keyword"},
        "start_time": {"type": "float"},
        "end_time": {"type": "float"},
        "duration": {"type": "float"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "caption": {"type": "text"},
        "tags": {"type": "keyword"},
        "language": {"type": "keyword"},
        "content_hash": {"type": "keyword"},
        "indexed_at": {
            "type": "date",
            "format": (
                "yyyy-MM-dd HH:mm:ss||strict_date_optional_time||epoch_millis"
            ),
        },
    }
    if dims > 0:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }
    return properties


def ensure_index() -> None:
    """创建索引（若不存在）；已存在时补充 project/filename 等新字段映射。"""
    client = get_client()
    name = index_name()
    dims = int(get_setting("clip_index_embedding_dims") or 0)
    properties = _index_properties(dims)

    if client.indices.exists(index=name):
        # 已有索引追加新字段（不会改已有字段类型）。
        try:
            client.indices.put_mapping(
                index=name,
                properties={
                    "project": properties["project"],
                    "filename": properties["filename"],
                    "tags": properties["tags"],
                    "indexed_at": properties["indexed_at"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"put_mapping for locales project fields failed: {exc}")
        return

    body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": properties},
    }
    try:
        client.indices.create(index=name, **body)
    except TypeError:
        client.indices.create(index=name, body=body)
    logger.info(f"created elasticsearch index: {name}")


def ping() -> bool:
    try:
        return bool(get_client().ping())
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"elasticsearch ping failed: {exc}")
        return False


def upsert_clip(document: dict[str, Any]) -> str:
    """按 clip_id 幂等写入一条分片理解文档。"""
    ensure_index()
    clip_id = str(document.get("clip_id") or "").strip()
    if not clip_id:
        raise ClipIndexError("clip_id is required")

    payload = dict(document)
    payload.setdefault("indexed_at", format_indexed_at())
    client = get_client()
    client.index(index=index_name(), id=clip_id, document=payload)
    return clip_id


def bulk_upsert_clips(documents: list[dict[str, Any]]) -> int:
    if not documents:
        return 0
    ensure_index()
    from elasticsearch.helpers import bulk

    now = format_indexed_at()
    actions = []
    for doc in documents:
        clip_id = str(doc.get("clip_id") or "").strip()
        if not clip_id:
            continue
        payload = dict(doc)
        payload.setdefault("indexed_at", now)
        actions.append(
            {
                "_op_type": "index",
                "_index": index_name(),
                "_id": clip_id,
                "_source": payload,
            }
        )
    if not actions:
        return 0
    success, _ = bulk(get_client(), actions, raise_on_error=False)
    return int(success)


def _source_query(
    source_path: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if source_path:
        filters.append({"term": {"source_path": source_path}})
    if project:
        filters.append({"term": {"project": project}})
    if not filters:
        return {"match_all": {}}
    if len(filters) == 1:
        return filters[0]
    return {"bool": {"filter": filters}}


def search_clips(
    query: str,
    *,
    size: int = 10,
    source_path: str | None = None,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """
    按自然语言检索分片。

    默认使用 BM25 检索 caption/tags/filename；可按 project 过滤。
    """
    ensure_index()
    q = (query or "").strip()
    if not q:
        return []

    must: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": q,
                "fields": [
                    "caption^3",
                    "tags^3",
                    "filename^2",
                    "source_name",
                    "project",
                ],
                "type": "best_fields",
            }
        }
    ]
    if source_path:
        must.append({"term": {"source_path": source_path}})
    if project:
        must.append({"term": {"project": project}})

    body = {
        "size": max(1, min(int(size), 100)),
        "query": {"bool": {"must": must}},
        "_source": {
            "excludes": ["embedding"],
        },
    }
    # elasticsearch-py 8 推荐扁平参数；同时兼容仍接受 body 的调用方式。
    try:
        resp = get_client().search(
            index=index_name(),
            query=body["query"],
            size=body["size"],
            source_excludes=["embedding"],
        )
    except TypeError:
        resp = get_client().search(index=index_name(), body=body)
    hits = resp.get("hits", {}).get("hits", [])
    results = []
    for hit in hits:
        item = dict(hit.get("_source") or {})
        item["_score"] = hit.get("_score")
        results.append(item)
    return results


def delete_by_source(
    source_path: str,
    project: str | None = None,
) -> int:
    """删除某源视频（可选按项目）下的全部分片文档。"""
    ensure_index()
    client = get_client()
    query = _source_query(source_path=source_path, project=project)
    try:
        resp = client.delete_by_query(
            index=index_name(),
            query=query,
            conflicts="proceed",
            refresh=True,
        )
    except TypeError:
        resp = client.delete_by_query(
            index=index_name(),
            body={"query": query},
            conflicts="proceed",
            refresh=True,
        )
    return int(resp.get("deleted") or 0)


def source_has_clips(
    source_path: str,
    project: str | None = None,
) -> bool:
    """判断某个本地源文件在指定项目下是否已有分片理解文档。"""
    ensure_index()
    client = get_client()
    query = _source_query(source_path=source_path, project=project)
    try:
        resp = client.count(index=index_name(), query=query)
    except TypeError:
        resp = client.count(index=index_name(), body={"query": query})
    return int(resp.get("count") or 0) > 0


def count_by_project(project: str | None = None) -> int:
    """统计某项目（或全库）下的分片文档数。"""
    ensure_index()
    client = get_client()
    if project:
        query = {"term": {"project": project}}
    else:
        query = {"match_all": {}}
    try:
        resp = client.count(index=index_name(), query=query)
    except TypeError:
        resp = client.count(index=index_name(), body={"query": query})
    return int(resp.get("count") or 0)

