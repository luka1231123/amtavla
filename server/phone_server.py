import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, send_file, jsonify, make_response, request
from flask_socketio import SocketIO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain.config import load_brain_config
from brain.memory.capture import capture_note
from brain.memory.catalog import MemoryCatalog
from brain.memory.context_engine import ContextEngine
from brain.memory.tagging import TagEngine

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UI_FILE = os.path.join(os.path.dirname(__file__), "ui.html")
EXPORT_ROOT = REPO_ROOT / "exports" / "memory"
MEMORY_CATALOG = None
MEMORY_CATALOG_LOCK = threading.Lock()
MAX_DEBUG_EVENTS = 500
command_store = None
response_store = None
command_updated_at = None
response_updated_at = None
server_started_at = time.time()
debug_events = []
debug_event_counter = 0


def _get_memory_catalog() -> MemoryCatalog:
    global MEMORY_CATALOG
    if MEMORY_CATALOG is not None:
        return MEMORY_CATALOG
    with MEMORY_CATALOG_LOCK:
        if MEMORY_CATALOG is None:
            config = load_brain_config()
            memory_config = config.get("memory", {})
            db_dir = memory_config.get("db_dir", "brain/db")
            db_path = os.environ.get("AMTAVLA_CATALOG_DB") or memory_config.get(
                "catalog_db_path",
                os.path.join(db_dir, "memory_catalog.db"),
            )
            MEMORY_CATALOG = MemoryCatalog(db_path)
    return MEMORY_CATALOG


def _memory_api_error(exc: Exception):
    if isinstance(exc, KeyError):
        return jsonify({"error": str(exc).strip("'")}), 404
    if isinstance(exc, (TypeError, ValueError)):
        return jsonify({"error": str(exc)}), 400
    app.logger.exception("Memory API operation failed")
    return jsonify({"error": "Memory operation failed"}), 500


def _store_command(text: str):
    global command_store, command_updated_at
    command_store = text
    command_updated_at = time.time()


def _store_response(text: str):
    global response_store, response_updated_at
    response_store = text
    response_updated_at = time.time()


def _iso_utc(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _age_seconds(ts):
    if ts is None:
        return None
    return round(time.time() - ts, 2)


def _debug_state():
    return {
        "command": command_store,
        "response": response_store,
        "pending": {
            "command": command_store is not None,
            "response": response_store is not None,
        },
        "command_updated_at": _iso_utc(command_updated_at),
        "response_updated_at": _iso_utc(response_updated_at),
        "command_age_seconds": _age_seconds(command_updated_at),
        "response_age_seconds": _age_seconds(response_updated_at),
        "uptime_seconds": round(time.time() - server_started_at, 2),
        "debug_event_count": len(debug_events),
    }


def _append_debug_event(event_type: str, payload: dict):
    global debug_event_counter
    debug_event_counter += 1
    event = {
        "id": debug_event_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": payload,
    }
    debug_events.append(event)
    if len(debug_events) > MAX_DEBUG_EVENTS:
        overflow = len(debug_events) - MAX_DEBUG_EVENTS
        del debug_events[:overflow]
    return event


@app.route("/")
def index():
    return send_file(UI_FILE)


# /memory and /debug are kept as deep-links into the unified UI (they open the
# matching tab via the URL hash) so old bookmarks keep working.
@app.route("/memory")
def memory_view():
    return _tab_redirect("memory")


@app.route("/debug")
def debug_view():
    return _tab_redirect("logs")


def _tab_redirect(tab: str):
    return make_response("", 302, {"Location": f"/#{tab}"})


@app.route("/api/memory/overview", methods=["GET"])
def memory_overview():
    return jsonify(_get_memory_catalog().overview())


@app.route("/api/memory/items", methods=["GET"])
def memory_items():
    try:
        rows = _get_memory_catalog().list_items(
            query=request.args.get("q", ""),
            item_type=request.args.get("type") or None,
            review_state=request.args.get("state") or None,
            include_deleted=request.args.get("include_deleted", "0") == "1",
            entity_id=(
                int(request.args["entity_id"])
                if request.args.get("entity_id")
                else None
            ),
            tag=request.args.get("tag") or None,
            since=float(request.args["since"]) if request.args.get("since") else None,
            until=float(request.args["until"]) if request.args.get("until") else None,
            limit=request.args.get("limit", 200),
            offset=request.args.get("offset", 0),
        )
        return jsonify({"items": rows, "count": len(rows)})
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/items/<int:item_id>", methods=["GET"])
def memory_item_detail(item_id: int):
    try:
        return jsonify(_get_memory_catalog().inspect_item(item_id))
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/items/<int:item_id>/correct", methods=["POST"])
def correct_memory_item(item_id: int):
    data = request.get_json(silent=True) or {}
    try:
        item = _get_memory_catalog().correct_item(
            item_id,
            data.get("content", ""),
            data.get("reason", ""),
        )
        return jsonify(item)
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/items/<int:item_id>/state", methods=["POST"])
def set_memory_item_state(item_id: int):
    data = request.get_json(silent=True) or {}
    try:
        item = _get_memory_catalog().set_review_state(
            item_id,
            data.get("review_state", ""),
            data.get("note", ""),
        )
        return jsonify(item)
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/merge", methods=["POST"])
def merge_memory_items():
    data = request.get_json(silent=True) or {}
    try:
        item = _get_memory_catalog().merge_items(
            int(data.get("target_id")),
            list(data.get("source_ids") or []),
            content=data.get("content") or None,
            note=data.get("note", ""),
        )
        return jsonify(item)
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/entities", methods=["GET", "POST"])
def memory_entities():
    catalog = _get_memory_catalog()
    try:
        if request.method == "GET":
            rows = catalog.list_entities(
                query=request.args.get("q", ""),
                limit=request.args.get("limit", 200),
            )
            return jsonify({"entities": rows, "count": len(rows)})
        data = request.get_json(silent=True) or {}
        entity = catalog.upsert_entity(
            data.get("entity_type", "other"),
            data.get("name", ""),
            description=data.get("description", ""),
            review_state=data.get("review_state", "candidate"),
            metadata=data.get("metadata") or {},
        )
        if data.get("memory_item_id") is not None:
            catalog.link_item_entity(
                int(data["memory_item_id"]),
                entity["id"],
                role=data.get("role", "related"),
                confidence=float(data.get("confidence", 0.5)),
            )
        return jsonify(entity), 201
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/relations", methods=["GET", "POST"])
def memory_relations():
    catalog = _get_memory_catalog()
    try:
        if request.method == "GET":
            rows = catalog.list_relations(limit=request.args.get("limit", 500))
            return jsonify({"relations": rows, "count": len(rows)})
        data = request.get_json(silent=True) or {}
        relation = catalog.create_relation(
            int(data.get("source_entity_id")),
            data.get("relation_type", ""),
            int(data.get("target_entity_id")),
            memory_item_id=data.get("memory_item_id"),
            confidence=float(data.get("confidence", 0.5)),
            metadata=data.get("metadata") or {},
        )
        return jsonify(relation), 201
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/items/<int:item_id>/tags", methods=["POST"])
def add_memory_item_tag(item_id: int):
    data = request.get_json(silent=True) or {}
    try:
        assignment = _get_memory_catalog().assign_tag(
            item_id,
            data.get("tag_type", ""),
            data.get("name", ""),
            status=data.get("status", "accepted"),
            confidence=float(data.get("confidence", 1.0)),
            source="user",
        )
        return jsonify(assignment), 201
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/items/<int:item_id>/tags/<int:tag_id>", methods=["POST"])
def review_memory_item_tag(item_id: int, tag_id: int):
    """One-tap tag review: accept, reject, or correct (with replacement)."""
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip().lower()
    catalog = _get_memory_catalog()
    try:
        if action in ("accepted", "accept"):
            result = catalog.set_tag_status(item_id, tag_id, "accepted", data.get("note", ""))
        elif action in ("rejected", "reject"):
            result = catalog.set_tag_status(item_id, tag_id, "rejected", data.get("note", ""))
        elif action in ("corrected", "correct"):
            result = catalog.correct_tag_assignment(
                item_id,
                tag_id,
                data.get("tag_type", ""),
                data.get("name", ""),
                note=data.get("note", ""),
            )
        else:
            raise ValueError(f"Unsupported tag action: {action}")
        return jsonify(result)
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/tags", methods=["GET"])
def memory_tags():
    try:
        rows = _get_memory_catalog().list_tags(
            tag_type=request.args.get("type") or None
        )
        return jsonify({"tags": rows, "count": len(rows)})
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/capture", methods=["POST"])
def capture_memory():
    """Direct capture pipeline: store a note, auto-tag it, extract commitments,
    log the capture event. Shares brain.memory.capture.capture_note with the
    in-process MemoryService so both entry points stay in sync."""
    data = request.get_json(silent=True) or {}
    catalog = _get_memory_catalog()
    try:
        content = (data.get("content") or "").strip()
        session_id = data.get("session_id", "capture")
        result = capture_note(
            catalog,
            TagEngine(catalog),
            ContextEngine(catalog),
            content,
            item_type=data.get("item_type", "episode"),
            capture_type=data.get("capture_type", "pasted"),
            session_id=session_id,
        )
        return jsonify(result), 201
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/api/memory/export", methods=["POST"])
def export_memory():
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_dir = EXPORT_ROOT / stamp
        result = _get_memory_catalog().export(str(export_dir))
        archive_path = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir)
        response = send_file(
            archive_path,
            as_attachment=True,
            download_name=f"amtavla-memory-{stamp}.zip",
            mimetype="application/zip",
        )
        response.headers["X-Memory-Item-Count"] = str(result["item_count"])
        return response
    except Exception as exc:
        return _memory_api_error(exc)


@app.route("/command", methods=["POST"])
def set_command():
    data = request.get_json()
    if data and "text" in data:
        _store_command(data["text"])
        socketio.emit("command_submitted", {"text": data["text"], "source": "http"})
        return jsonify({"status": "ok"})
    return jsonify({"error": "No text provided"}), 400


@app.route("/command", methods=["GET"])
def get_command():
    global command_store
    text = command_store
    return jsonify({"command": text})


@app.route("/command/ack", methods=["POST"])
def ack_command():
    global command_store, command_updated_at
    command_store = None
    command_updated_at = time.time()
    return jsonify({"status": "ok"})


@app.route("/response", methods=["POST"])
def set_response():
    data = request.get_json()
    if data and "text" in data:
        _store_response(data["text"])
        socketio.emit("assistant_response", {"text": data["text"], "source": "http"})
        return jsonify({"status": "ok"})
    return jsonify({"error": "No text provided"}), 400


@app.route("/response", methods=["GET"])
def get_response():
    global response_store
    text = response_store
    return jsonify({"response": text})


@app.route("/response/ack", methods=["POST"])
def ack_response():
    global response_store, response_updated_at
    response_store = None
    response_updated_at = time.time()
    return jsonify({"status": "ok"})


@app.route("/debug/state", methods=["GET"])
def debug_state():
    return jsonify(_debug_state())


@app.route("/debug/event", methods=["POST"])
def push_debug_event():
    data = request.get_json(silent=True) or {}
    event_type = data.get("type", "event")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {"value": payload}
    event = _append_debug_event(str(event_type), payload)
    socketio.emit("debug_event", event)
    return jsonify({"status": "ok", "event_id": event["id"]})


@app.route("/debug/events", methods=["GET"])
def get_debug_events():
    limit_arg = request.args.get("limit")
    since_arg = request.args.get("since")

    limit = 200
    if limit_arg is not None:
        try:
            limit = max(1, min(MAX_DEBUG_EVENTS, int(limit_arg)))
        except ValueError:
            limit = 200

    since_id = 0
    if since_arg is not None:
        try:
            since_id = max(0, int(since_arg))
        except ValueError:
            since_id = 0

    filtered = [e for e in debug_events if e["id"] > since_id]
    out = filtered[-limit:]
    return jsonify(
        {
            "events": out,
            "count": len(out),
            "total": len(debug_events),
            "latest_id": out[-1]["id"] if out else since_id,
        }
    )


@app.route("/debug/events/clear", methods=["POST"])
def clear_debug_events():
    global debug_event_counter
    debug_events.clear()
    debug_event_counter = 0
    return jsonify({"status": "ok"})



@socketio.on("submit_command")
def ws_submit_command(data):
    text = ((data or {}).get("text") or "").strip()
    if not text:
        return {"status": "error", "message": "No text provided"}
    _store_command(text)
    socketio.emit("command_submitted", {"text": text, "source": "socket"})
    return {"status": "ok"}


@socketio.on("assistant_response")
def ws_assistant_response(data):
    text = ((data or {}).get("text") or "").strip()
    if not text:
        return {"status": "error", "message": "No text provided"}
    _store_response(text)
    socketio.emit("assistant_response", {"text": text, "source": "socket"})
    return {"status": "ok"}


@socketio.on("memory_check")
def ws_memory_check(data):
    payload = data or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return {"status": "error", "message": "No text provided"}
    socketio.emit(
        "memory_check",
        {"text": text, "insight_id": payload.get("insight_id"), "source": "socket"},
    )
    return {"status": "ok"}


@socketio.on("insight_feedback")
def ws_insight_feedback(data):
    payload = data or {}
    if payload.get("insight_id") is None:
        return {"status": "error", "message": "No insight_id provided"}
    socketio.emit(
        "insight_feedback",
        {
            "insight_id": payload.get("insight_id"),
            "keep": bool(payload.get("keep")),
            "source": "socket",
        },
    )
    return {"status": "ok"}


@socketio.on("debug_event")
def ws_debug_event(data):
    data = data or {}
    event_type = data.get("type", "event")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {"value": payload}
    event = _append_debug_event(str(event_type), payload)
    socketio.emit("debug_event", event)
    return {"status": "ok", "event_id": event["id"]}


if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=8081,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
