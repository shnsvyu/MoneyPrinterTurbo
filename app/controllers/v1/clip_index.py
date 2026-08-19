"""本地视频分片理解索引 API。"""

from fastapi import BackgroundTasks, Request
from loguru import logger

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import ClipIndexRequest, ClipSearchRequest
from app.services import clip_es, clip_index
from app.utils import utils

router = new_router()


def _require_enabled(request_id: str) -> None:
    if not clip_index.is_enabled():
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=(
                "clip index is disabled; set clip_index_enabled=true "
                "and configure elasticsearch_hosts in config.toml"
            ),
        )


@router.get(
    "/clip-index/health",
    summary="Check clip index / Elasticsearch connectivity",
)
def clip_index_health(request: Request):
    request_id = base.get_task_id(request)
    enabled = clip_index.is_enabled()
    reachable = False
    error = None
    if enabled:
        try:
            reachable = clip_es.ping()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return utils.get_response(
        200,
        {
            "enabled": enabled,
            "elasticsearch_reachable": reachable,
            "index_name": clip_es.index_name() if enabled else None,
            "error": error,
        },
    )


@router.post(
    "/clip-index/index",
    summary="Index local video clip understandings into Elasticsearch",
)
def index_local_clips(
    request: Request,
    body: ClipIndexRequest,
    background_tasks: BackgroundTasks,
):
    """
    同步索引单个视频；index_all=true 时在后台扫描 local_videos。

    大批量目录扫描可能较慢，因此 index_all 走 BackgroundTasks，立即返回 accepted。
    """
    request_id = base.get_task_id(request)
    _require_enabled(request_id)

    try:
        if body.index_all:
            project = body.project or ""
            background_tasks.add_task(
                clip_index.index_all_local_videos,
                project=project,
                max_clip_duration=body.max_clip_duration,
                clip_speed=body.clip_speed,
                force=body.force,
            )
            return utils.get_response(
                200,
                {
                    "accepted": True,
                    "mode": "index_all",
                    "project": project,
                    "message": "indexing all local videos in background",
                },
            )

        if not (body.video_path or "").strip():
            raise HttpException(
                task_id=request_id,
                status_code=400,
                message="video_path is required when index_all is false",
            )

        result = clip_index.index_local_video(
            body.video_path.strip(),
            project=body.project or "",
            filename=body.filename or "",
            max_clip_duration=body.max_clip_duration,
            clip_speed=body.clip_speed,
            force=body.force,
        )
        return utils.get_response(200, result)
    except HttpException:
        raise
    except clip_index.ClipUnderstandError as exc:
        raise HttpException(
            task_id=request_id, status_code=400, message=str(exc)
        ) from exc
    except clip_es.ClipIndexError as exc:
        raise HttpException(
            task_id=request_id, status_code=503, message=str(exc)
        ) from exc
    except ValueError as exc:
        raise HttpException(
            task_id=request_id, status_code=400, message=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("clip index failed")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"clip index failed: {exc}",
        ) from exc


@router.post(
    "/clip-index/search",
    summary="Search indexed local video clips by natural language",
)
def search_local_clips(request: Request, body: ClipSearchRequest):
    request_id = base.get_task_id(request)
    _require_enabled(request_id)
    try:
        if body.hybrid and (body.video_script or body.video_terms or body.query):
            hits = clip_index.search_local_clips_hybrid(
                query=body.query,
                video_script=body.video_script,
                video_terms=body.video_terms,
                project=body.project or "",
                size=body.size,
            )
            mode = "hybrid"
        else:
            query = (body.query or body.video_script or "").strip()
            if not query:
                raise HttpException(
                    task_id=request_id,
                    status_code=400,
                    message="query or video_script is required",
                )
            hits = clip_index.search_local_clips(
                query, size=body.size, project=body.project or ""
            )
            mode = "query"
        return utils.get_response(
            200,
            {
                "mode": mode,
                "query": body.query,
                "project": body.project or "",
                "hits": hits,
            },
        )
    except HttpException:
        raise
    except clip_index.ClipUnderstandError as exc:
        raise HttpException(
            task_id=request_id, status_code=400, message=str(exc)
        ) from exc
    except clip_es.ClipIndexError as exc:
        raise HttpException(
            task_id=request_id, status_code=503, message=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("clip search failed")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"clip search failed: {exc}",
        ) from exc
