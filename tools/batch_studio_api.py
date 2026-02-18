#!/usr/bin/env python3
"""Batch Studio API

Minimal remote API wrapper for headless catalog processing and report ingestion.

Endpoints:
  GET  /health
  POST /v1/catalog/process
  POST /v1/reports/ingest
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _load_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _append_jsonl(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _is_path_contained(user_path: str, allowed_root: pathlib.Path) -> bool:
    """Return True if *user_path* resolves inside *allowed_root*.

    Prevents directory-traversal attacks when the API is network-exposed.
    """
    try:
        resolved = pathlib.Path(user_path).resolve()
        root_resolved = allowed_root.resolve()
        return resolved == root_resolved or resolved.is_relative_to(root_resolved)
    except (ValueError, OSError):
        return False


def make_handler(
    *,
    automix_bin: str,
    output_root: pathlib.Path,
    api_key: str,
) -> type[BaseHTTPRequestHandler]:
    class BatchStudioHandler(BaseHTTPRequestHandler):
        server_version = "AutoMixBatchStudio/1.0"

        def _authorized(self) -> bool:
            if not api_key:
                return True
            provided = self.headers.get("x-api-key", "")
            return bool(provided) and provided == api_key

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json_response(self, HTTPStatus.OK, {"ok": True, "service": "batch-studio-api"})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                _json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return

            if self.path == "/v1/catalog/process":
                self._handle_catalog_process()
                return

            if self.path == "/v1/reports/ingest":
                self._handle_report_ingest()
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _handle_catalog_process(self) -> None:
            payload = _load_json_body(self)
            input_dir = payload.get("input")
            output_dir = payload.get("output")
            if not input_dir or not output_dir:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "input and output are required"},
                )
                return

            if not _is_path_contained(str(input_dir), output_root):
                _json_response(
                    self,
                    HTTPStatus.FORBIDDEN,
                    {"error": "input directory is outside the allowed root"},
                )
                return
            if not _is_path_contained(str(output_dir), output_root):
                _json_response(
                    self,
                    HTTPStatus.FORBIDDEN,
                    {"error": "output directory is outside the allowed root"},
                )
                return

            run_id = f"run_{int(time.time())}"
            run_dir = output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            csv_path = run_dir / "catalog_deliverables.csv"
            json_path = run_dir / "catalog_deliverables.json"
            checkpoint_path = run_dir / "catalog_checkpoint.json"

            cmd = [
                automix_bin,
                "catalog-process",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--checkpoint",
                str(checkpoint_path),
                "--csv",
                str(csv_path),
                "--json",
                str(json_path),
            ]

            optional_map = {
                "renderer": "--renderer",
                "format": "--format",
                "analysis_threads": "--analysis-threads",
                "render_parallelism": "--render-parallelism",
            }
            for key, flag in optional_map.items():
                value = payload.get(key)
                if value is not None and value != "":
                    cmd.extend([flag, str(value)])

            if payload.get("resume"):
                cmd.append("--resume")

            proc = subprocess.run(cmd, capture_output=True, text=True)

            response_payload: Dict[str, Any] = {
                "ok": proc.returncode == 0,
                "run_id": run_id,
                "exit_code": proc.returncode,
                "command": cmd,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
                "json_report": str(json_path),
                "csv_report": str(csv_path),
                "checkpoint": str(checkpoint_path),
            }
            if json_path.exists():
                try:
                    response_payload["deliverables"] = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    response_payload["deliverables"] = {"error": "failed_to_parse_deliverables"}

            _append_jsonl(
                output_root / "runs" / "catalog_process_runs.jsonl",
                {
                    "timestamp": int(time.time()),
                    "run_id": run_id,
                    "ok": response_payload["ok"],
                    "exit_code": response_payload["exit_code"],
                    "json_report": response_payload["json_report"],
                    "csv_report": response_payload["csv_report"],
                },
            )

            status = HTTPStatus.OK if proc.returncode == 0 else HTTPStatus.BAD_GATEWAY
            _json_response(self, status, response_payload)

        def _handle_report_ingest(self) -> None:
            payload = _load_json_body(self)
            report_obj = payload.get("report")
            report_path = payload.get("report_path")

            if report_obj is None and not report_path:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "report or report_path is required"})
                return

            record: Dict[str, Any] = {
                "ingested_at_epoch": int(time.time()),
                "source": payload.get("source", "manual"),
            }

            if report_obj is not None:
                record["report"] = report_obj

            if report_path:
                if not _is_path_contained(str(report_path), output_root):
                    _json_response(
                        self,
                        HTTPStatus.FORBIDDEN,
                        {"error": "report_path is outside the allowed root"},
                    )
                    return
                path = pathlib.Path(str(report_path))
                record["report_path"] = str(path)
                if path.exists():
                    try:
                        record["report"] = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        record["report_parse_error"] = True
                else:
                    record["report_missing"] = True

            ingest_log = output_root / "ingested" / "reports.jsonl"
            _append_jsonl(ingest_log, record)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "ingested_log": str(ingest_log),
                    "ingested_at_epoch": record["ingested_at_epoch"],
                },
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep logs concise for headless usage.
            sys.stderr.write("[batch-studio-api] " + (fmt % args) + "\n")

    return BatchStudioHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoMixMaster Batch Studio API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8089, help="Bind port")
    parser.add_argument(
        "--automix-bin",
        default=os.environ.get("AUTOMIX_DEV_TOOLS_BIN", "automix_dev_tools"),
        help="Path to automix_dev_tools binary",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("AUTOMIX_BATCH_API_ROOT", "artifacts/batch_studio_api"),
        help="Directory for API run artifacts",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AUTOMIX_BATCH_API_KEY", ""),
        help="Optional API key; if provided, clients must pass x-api-key",
    )
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    handler = make_handler(
        automix_bin=args.automix_bin,
        output_root=output_root,
        api_key=args.api_key,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Batch Studio API listening on http://{args.host}:{args.port}")
    print(f"Using automix_dev_tools binary: {args.automix_bin}")
    print(f"Artifacts root: {output_root}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
