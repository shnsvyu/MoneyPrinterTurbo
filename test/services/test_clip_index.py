import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os

from app.config import config
from app.models.schema import VideoConcatMode
from app.services import clip_es, clip_index, video


class TestBuildSubclippedItems(unittest.TestCase):
    def _fake_clip(self, duration=12.0, size=(640, 480)):
        clip = MagicMock()
        clip.duration = duration
        clip.size = size
        return clip

    @patch.object(video, "close_clip")
    @patch.object(video, "_open_video_clip_quietly")
    def test_random_mode_slices_full_timeline(self, open_clip, _close):
        open_clip.return_value = self._fake_clip(duration=12.0)
        items = video.build_subclipped_items(
            video_paths=["/tmp/sample.mp4"],
            max_clip_duration=5,
            clip_speed=1.0,
            video_concat_mode=VideoConcatMode.random,
            prioritize_unique=False,
        )
        self.assertEqual(len(items), 3)
        self.assertEqual(
            [(i.start_time, i.end_time) for i in items],
            [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)],
        )

    @patch.object(video, "close_clip")
    @patch.object(video, "_open_video_clip_quietly")
    def test_sequential_mode_keeps_first_segment_only(self, open_clip, _close):
        open_clip.return_value = self._fake_clip(duration=12.0)
        items = video.build_subclipped_items(
            video_paths=["/tmp/sample.mp4"],
            max_clip_duration=5,
            clip_speed=1.0,
            video_concat_mode=VideoConcatMode.sequential,
            prioritize_unique=False,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].start_time, 0.0)
        self.assertEqual(items[0].end_time, 5.0)

    @patch.object(video, "close_clip")
    @patch.object(video, "_open_video_clip_quietly")
    def test_clip_speed_scales_source_window(self, open_clip, _close):
        open_clip.return_value = self._fake_clip(duration=12.0)
        items = video.build_subclipped_items(
            video_paths=["/tmp/sample.mp4"],
            max_clip_duration=3,
            clip_speed=2.0,
            video_concat_mode=VideoConcatMode.random,
            prioritize_unique=False,
        )
        # 成片 3s * 2x 速度 => 源窗口 6s
        self.assertEqual(items[0].end_time - items[0].start_time, 6.0)


class TestClipIndexHelpers(unittest.TestCase):
    def test_parse_understanding_json(self):
        parsed = clip_index._parse_understanding_json(
            '{"caption":"城市夜景","tags":["城市","夜景"],"language":"zh"}'
        )
        self.assertEqual(parsed["caption"], "城市夜景")
        self.assertEqual(parsed["tags"], ["城市", "夜景"])

    def test_clip_document_id_stable(self):
        a = clip_index.clip_document_id("/a/b.mp4", 0.0, 5.0)
        b = clip_index.clip_document_id("/a/b.mp4", 0.0, 5.0)
        c = clip_index.clip_document_id("/a/b.mp4", 5.0, 10.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_is_enabled_follows_config(self):
        previous_app = config.app.get("clip_index_enabled")
        previous_es = config.es.get("clip_index_enabled")
        try:
            config.es["clip_index_enabled"] = False
            config.app["clip_index_enabled"] = False
            self.assertFalse(clip_index.is_enabled())
            config.es["clip_index_enabled"] = True
            self.assertTrue(clip_index.is_enabled())
        finally:
            if previous_es is None:
                config.es.pop("clip_index_enabled", None)
            else:
                config.es["clip_index_enabled"] = previous_es
            if previous_app is None:
                config.app.pop("clip_index_enabled", None)
            else:
                config.app["clip_index_enabled"] = previous_app


class TestClipEsDisabled(unittest.TestCase):
    def test_search_requires_enabled(self):
        previous_app = config.app.get("clip_index_enabled")
        previous_es = config.es.get("clip_index_enabled")
        try:
            config.es["clip_index_enabled"] = False
            config.app["clip_index_enabled"] = False
            with self.assertRaises(clip_index.ClipUnderstandError):
                clip_index.search_local_clips("城市")
        finally:
            if previous_es is None:
                config.es.pop("clip_index_enabled", None)
            else:
                config.es["clip_index_enabled"] = previous_es
            if previous_app is None:
                config.app.pop("clip_index_enabled", None)
            else:
                config.app["clip_index_enabled"] = previous_app

    def test_bulk_upsert_empty(self):
        self.assertEqual(clip_es.bulk_upsert_clips([]), 0)

    @patch.object(clip_es, "get_client")
    @patch.object(clip_es, "ensure_index")
    def test_search_clips_maps_hits(self, _ensure, get_client):
        client = MagicMock()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 1.5,
                        "_source": {
                            "clip_id": "abc",
                            "caption": "城市夜景",
                            "source_path": "/x.mp4",
                        },
                    }
                ]
            }
        }
        get_client.return_value = client
        hits = clip_es.search_clips("城市", size=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["caption"], "城市夜景")
        self.assertEqual(hits[0]["_score"], 1.5)


class TestHybridRanking(unittest.TestCase):
    def test_split_script_units(self):
        units = clip_index.split_script_units(
            "城市醒来了。阳光洒在街道上！\n咖啡店开门迎客。"
        )
        self.assertGreaterEqual(len(units), 3)

    def test_select_clips_for_duration(self):
        ranked = [
            {"clip_id": "a", "duration": 5},
            {"clip_id": "b", "duration": 5},
            {"clip_id": "c", "duration": 5},
        ]
        selected = clip_index.select_clips_for_duration(
            ranked, audio_duration=9.0, max_clip_duration=5
        )
        self.assertEqual(len(selected), 2)

    @patch.object(clip_es, "search_clips")
    def test_hybrid_rank_sentence_order_first(self, search_clips):
        def _fake_search(query, size=10, source_path=None, project=None):
            if "第一句" in query:
                return [
                    {
                        "clip_id": "s1",
                        "source_path": "/a.mp4",
                        "caption": "第一句画面",
                        "duration": 5,
                        "_score": 10,
                    }
                ]
            if "第二句" in query:
                return [
                    {
                        "clip_id": "s2",
                        "source_path": "/b.mp4",
                        "caption": "第二句画面",
                        "duration": 5,
                        "_score": 9,
                    }
                ]
            if query == "城市":
                return [
                    {
                        "clip_id": "k1",
                        "source_path": "/c.mp4",
                        "caption": "城市关键词",
                        "duration": 5,
                        "_score": 8,
                    }
                ]
            return []

        search_clips.side_effect = _fake_search
        previous = None
        try:
            # force enabled path through hybrid_rank_clips
            with patch.object(clip_index, "is_enabled", return_value=True):
                ranked = clip_index.hybrid_rank_clips(
                    video_script="第一句。第二句。",
                    video_terms=["城市"],
                    size_per_query=5,
                )
            ids = [c["clip_id"] for c in ranked]
            self.assertEqual(ids[:2], ["s1", "s2"])
            self.assertIn("k1", ids)
            self.assertIn("sentence", ranked[0]["_match_sources"])
        finally:
            pass


class TestLocalesVideoSourceHelpers(unittest.TestCase):
    def test_helpers(self):
        from app.models import const

        self.assertTrue(const.is_local_video_source("local"))
        self.assertTrue(const.is_local_video_source("locales"))
        self.assertTrue(const.is_locales_video_source("locales"))
        self.assertFalse(const.is_locales_video_source("local"))
        self.assertTrue(const.is_supported_video_source("locales"))
        self.assertFalse(const.is_supported_video_source("unknown"))

    def test_normalize_locales_project(self):
        from app.models import const

        self.assertEqual(
            const.normalize_locales_project("国际新闻"), "国际新闻"
        )
        self.assertEqual(const.normalize_locales_project("电影"), "电影")
        self.assertEqual(const.normalize_locales_project("仙逆"), "仙逆")
        self.assertEqual(
            const.normalize_locales_project("凡人修仙传"), "凡人修仙传"
        )
        # 兼容笔误「凡人修仙转」
        self.assertEqual(
            const.normalize_locales_project("凡人修仙转"), "凡人修仙传"
        )
        self.assertEqual(
            const.normalize_locales_project(""), const.DEFAULT_LOCALES_PROJECT
        )
        self.assertTrue(const.is_supported_locales_project("凡人修仙转"))


class TestBuildClipDocumentProjectFields(unittest.TestCase):
    def test_document_contains_project_filename_tags_times(self):
        item = video.SubClippedVideoClip(
            file_path="/tmp/demo.mp4",
            source_file_path="/data/films/material-abc.mp4",
            start_time=12.5,
            end_time=17.5,
            duration=5.0,
            width=1920,
            height=1080,
        )
        doc = clip_index.build_clip_document(
            item,
            understanding={"caption": "主角修炼", "tags": ["修仙", "飞升"]},
            content_hash="abc",
            project="仙逆",
            filename="仙逆第1集.mp4",
        )
        self.assertEqual(doc["project"], "仙逆")
        self.assertEqual(doc["filename"], "仙逆第1集.mp4")
        self.assertEqual(doc["source_name"], "material-abc.mp4")
        self.assertEqual(doc["tags"], ["修仙", "飞升"])
        self.assertNotIn("keywords", doc)
        self.assertEqual(doc["start_time"], 12.5)
        self.assertEqual(doc["end_time"], 17.5)
        self.assertEqual(len(doc["clip_id"]), 40)
        self.assertEqual(
            doc["clip_id"],
            clip_index.clip_document_id(
                "/data/films/material-abc.mp4", 12.5, 17.5, project="仙逆"
            ),
        )

    def test_resolve_display_filename_prefers_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "material-uuid.mp4")
            Path(path).write_bytes(b"x")
            clip_index.write_original_filename_meta(path, "凡人修仙传-01.mp4")
            self.assertEqual(
                clip_index.resolve_display_filename(path),
                "凡人修仙传-01.mp4",
            )


class TestIndexedAtFormat(unittest.TestCase):
    def test_format_indexed_at(self):
        value = clip_es.format_indexed_at()
        self.assertRegex(value, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class TestVisionRateLimitHelpers(unittest.TestCase):
    def test_is_rate_limit_error(self):
        self.assertTrue(
            clip_index._is_rate_limit_error(
                Exception("Error code: 429 - rate_limit_reached_error")
            )
        )
        self.assertFalse(clip_index._is_rate_limit_error(Exception("temperature")))

    def test_retry_after_seconds_parses_message(self):
        delay = clip_index._retry_after_seconds(
            Exception("please try again after 1 seconds"), attempt=0
        )
        self.assertEqual(delay, 1.0)


if __name__ == "__main__":
    unittest.main()



