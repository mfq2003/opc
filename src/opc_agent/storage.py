"""本模块封装 OPC Agent 的 SQLite 运行、切片、预测、指标和 API 失败记录。

输入为已校验的数据模型和 JSON 可序列化元数据，输出为幂等持久化记录与可恢复的运行索引。
关键依赖为 Python sqlite3；数据库仅保存路径、哈希和结构化结果，不保存大图或模型内容。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .models import AgentPrediction, LayoutClip, OpcMetrics


class ExperimentStore:
    """负责初始化 SQLite 模式，并以主键去重方式写入实验元数据。"""

    def __init__(self, database: Path):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clips (
              clip_id TEXT PRIMARY KEY, parent_layout TEXT NOT NULL, split TEXT,
              payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
              run_id TEXT NOT NULL, clip_id TEXT NOT NULL, stage TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, clip_id, stage),
              FOREIGN KEY (run_id) REFERENCES runs(run_id), FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            );
            CREATE TABLE IF NOT EXISTS predictions (
              run_id TEXT NOT NULL, clip_id TEXT NOT NULL, stage TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, clip_id, stage),
              FOREIGN KEY (run_id) REFERENCES runs(run_id), FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            );
            CREATE TABLE IF NOT EXISTS qwen_failures (
              run_id TEXT NOT NULL, clip_id TEXT NOT NULL, response_sha256 TEXT NOT NULL,
              reason TEXT NOT NULL, PRIMARY KEY (run_id, clip_id, response_sha256)
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def upsert_run(self, run_id: str, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        """幂等写入唯一运行的配置和环境快照。"""
        self.connection.execute(
            "INSERT OR IGNORE INTO runs(run_id, config_json, metadata_json) VALUES (?, ?, ?)",
            (run_id, self._json(dict(config)), self._json(dict(metadata))),
        )
        self.connection.commit()

    def upsert_clip(self, clip: LayoutClip) -> None:
        """按 clip_id 幂等写入切片元数据，避免中断恢复时重复。"""
        self.connection.execute(
            "INSERT OR IGNORE INTO clips(clip_id, parent_layout, split, payload_json) VALUES (?, ?, ?, ?)",
            (clip.clip_id, clip.parent_layout, clip.split, self._json(clip.dict())),
        )
        self.connection.commit()

    def upsert_metrics(self, run_id: str, clip_id: str, stage: str, metrics: OpcMetrics) -> None:
        """按运行、切片和阶段写入一条指标记录。"""
        self.connection.execute(
            "INSERT OR REPLACE INTO metrics(run_id, clip_id, stage, payload_json) VALUES (?, ?, ?, ?)",
            (run_id, clip_id, stage, self._json(metrics.dict())),
        )
        self.connection.commit()

    def upsert_prediction(self, run_id: str, stage: str, prediction: AgentPrediction) -> None:
        """写入快速模型或闭环路由预测。"""
        self.connection.execute(
            "INSERT OR REPLACE INTO predictions(run_id, clip_id, stage, payload_json) VALUES (?, ?, ?, ?)",
            (run_id, prediction.clip_id, stage, self._json(prediction.dict())),
        )
        self.connection.commit()

    def record_qwen_failure(self, run_id: str, clip_id: str, response_sha256: str, reason: str) -> None:
        """记录无效 Qwen 响应的哈希和失败原因，原始响应由运行目录单独保存。"""
        self.connection.execute(
            "INSERT OR IGNORE INTO qwen_failures(run_id, clip_id, response_sha256, reason) VALUES (?, ?, ?, ?)",
            (run_id, clip_id, response_sha256, reason),
        )
        self.connection.commit()

    def count(self, table: str) -> int:
        """返回白名单表的记录数，供测试与报告使用。"""
        if table not in {"runs", "clips", "metrics", "predictions", "qwen_failures"}:
            raise ValueError("不允许查询未知表")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def close(self) -> None:
        """关闭数据库连接。"""
        self.connection.close()

