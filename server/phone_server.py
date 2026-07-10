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
from brain.memory.catalog import MemoryCatalog
from brain.memory.context_engine import ContextEngine
from brain.memory.tagging import TagEngine

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UI_FILE = os.path.join(os.path.dirname(__file__), "phone_ui.html")
MEMORY_UI_FILE = os.path.join(os.path.dirname(__file__), "memory_ui.html")
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


@app.route("/memory")
def memory_view():
    return send_file(MEMORY_UI_FILE)


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
    """Direct capture pipeline: store a note, auto-tag it, log the capture event."""
    data = request.get_json(silent=True) or {}
    catalog = _get_memory_catalog()
    try:
        content = (data.get("content") or "").strip()
        item = catalog.upsert_item(
            item_type=data.get("item_type", "episode"),
            content=content,
            review_state="confirmed",
            confidence=0.7,
            importance=0.5,
            metadata={"captured": True, "session_id": data.get("session_id", "capture")},
        )
        event = catalog.record_capture_event(
            data.get("capture_type", "pasted"),
            content,
            session_id=data.get("session_id", "capture"),
            memory_item_id=item["id"],
        )
        engine = TagEngine(catalog)
        context = ContextEngine(catalog).current_context(
            data.get("session_id", "capture")
        )
        tags = engine.tag_item(item, session_context=context)
        return jsonify(
            {"item": catalog.inspect_item(item["id"]), "event": event, "tags": tags}
        ), 201
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


@app.route("/debug")
def debug_view():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debug - amtavla</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, monospace;
            background: #1a1a1a; color: #e0e0e0; padding: 16px;
        }
        h1 { color: #4fc3f7; margin-bottom: 16px; }
        .card { 
            background: #2a2a2a; border-radius: 8px; padding: 16px; margin-bottom: 16px;
        }
        .label { color: #888; font-size: 12px; margin-bottom: 4px; }
        .value { font-size: 16px; white-space: pre-wrap; }
        .ok { color: #4caf50; font-weight: 700; }
        .warn { color: #ff9800; font-weight: 700; }
        .bad { color: #ef5350; font-weight: 700; }
        .empty { color: #666; font-style: italic; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        .toolbar {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        button {
            border: none;
            background: #1565c0;
            color: #fff;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 600;
            cursor: pointer;
        }
        .log {
            max-height: 45vh;
            overflow-y: auto;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            background: #111;
            padding: 8px;
        }
        .event {
            border-bottom: 1px solid #2a2a2a;
            padding: 10px;
        }
        .event:last-child {
            border-bottom: none;
        }
        .event-meta {
            color: #8cc7ff;
            font-size: 12px;
            margin-bottom: 6px;
        }
        .event pre {
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 12px;
            line-height: 1.35;
            color: #d6d6d6;
        }
    </style>
</head>
<body>
    <h1>amtavla Debug</h1>
    <div class="toolbar">
        <button id="clearEvents">Clear Brain Log</button>
    </div>
    <div class="card">
        <div class="label">Overall Health</div>
        <div class="value" id="health"></div>
    </div>
    <div class="grid">
        <div class="card">
            <div class="label">Command Store</div>
            <div class="value" id="command"></div>
        </div>
        <div class="card">
            <div class="label">Response Store</div>
            <div class="value" id="response"></div>
        </div>
        <div class="card">
            <div class="label">Command Age (s)</div>
            <div class="value" id="commandAge"></div>
        </div>
        <div class="card">
            <div class="label">Response Age (s)</div>
            <div class="value" id="responseAge"></div>
        </div>
        <div class="card">
            <div class="label">Server Uptime (s)</div>
            <div class="value" id="uptime"></div>
        </div>
        <div class="card">
            <div class="label">Last Command Update</div>
            <div class="value" id="commandUpdated"></div>
        </div>
        <div class="card">
            <div class="label">Last Response Update</div>
            <div class="value" id="responseUpdated"></div>
        </div>
        <div class="card">
            <div class="label">Brain Event Count</div>
            <div class="value" id="eventCount"></div>
        </div>
    </div>
    <div class="card">
        <div class="label">Brain Events (prompt/context/plan/search/answer)</div>
        <div id="eventLog" class="log"></div>
    </div>
    <div class="card">
        <div class="label">Status</div>
        <div class="value" id="status">Waiting...</div>
    </div>
    <script src="/socket.io/socket.io.js"></script>
    <script>
        const STALE_SECONDS = 30;
        const MAX_EVENTS = 500;
        const state = {
            command: null,
            response: null,
            command_updated_at: null,
            response_updated_at: null,
            command_age_seconds: null,
            response_age_seconds: null,
            uptime_seconds: null,
            debug_event_count: 0,
            pending: { command: false, response: false },
        };
        let events = [];

        function setValue(id, value, emptyText = '(empty)') {
            const el = document.getElementById(id);
            const hasValue = value !== null && value !== undefined && value !== '';
            el.textContent = hasValue ? value : emptyText;
            el.className = hasValue ? 'value' : 'value empty';
        }

        function escapeHtml(value) {
            const text = String(value ?? '');
            return text
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;');
        }

        function secondsSince(isoTs) {
            if (!isoTs) return null;
            const ms = Date.parse(isoTs);
            if (Number.isNaN(ms)) return null;
            return Math.max(0, ((Date.now() - ms) / 1000));
        }

        function recomputeDerived() {
            const commandAge = secondsSince(state.command_updated_at);
            const responseAge = secondsSince(state.response_updated_at);
            state.command_age_seconds = commandAge === null ? null : Number(commandAge.toFixed(2));
            state.response_age_seconds = responseAge === null ? null : Number(responseAge.toFixed(2));
            if (state.uptime_seconds !== null) {
                state.uptime_seconds = Number((state.uptime_seconds + 1).toFixed(2));
            }
        }

        function renderState() {
            setValue('command', state.command);
            setValue('response', state.response);
            setValue('commandAge', state.command_age_seconds, 'n/a');
            setValue('responseAge', state.response_age_seconds, 'n/a');
            setValue('uptime', state.uptime_seconds, 'n/a');
            setValue('commandUpdated', state.command_updated_at, 'n/a');
            setValue('responseUpdated', state.response_updated_at, 'n/a');
            setValue('eventCount', state.debug_event_count, '0');

            const commandStale = state.pending.command && state.command_age_seconds !== null && state.command_age_seconds > STALE_SECONDS;
            const responseStale = state.pending.response && state.response_age_seconds !== null && state.response_age_seconds > STALE_SECONDS;
            const health = document.getElementById('health');
            if (commandStale || responseStale) {
                health.textContent = 'STALE';
                health.className = 'value warn';
            } else {
                health.textContent = 'OK';
                health.className = 'value ok';
            }
        }

        function renderEvents() {
            const log = document.getElementById('eventLog');
            if (!events.length) {
                log.innerHTML = '<div class="event"><div class="value empty">No events yet.</div></div>';
                return;
            }
            log.innerHTML = events.map((event) => {
                const payload = event.payload || {};
                const summary = payload.summary || '(no summary)';
                const details = JSON.stringify(payload.data || payload, null, 2);
                return `
                    <div class="event">
                        <div class="event-meta">#${event.id} | ${event.timestamp} | ${event.type}</div>
                        <div class="value" style="margin-bottom:8px; font-size:13px; color:#9fe3a7;">${escapeHtml(summary)}</div>
                        <pre>${escapeHtml(details)}</pre>
                    </div>
                `;
            }).join('');
        }

        function applyStatePatch(data) {
            if (!data || typeof data !== 'object') return;
            Object.assign(state, data);
            if (!state.pending) {
                state.pending = { command: Boolean(state.command), response: Boolean(state.response) };
            }
            state.debug_event_count = Number(state.debug_event_count || events.length || 0);
            renderState();
        }

        function appendEvent(event) {
            if (!event || typeof event !== 'object') return;
            events.push(event);
            if (events.length > MAX_EVENTS) {
                events = events.slice(events.length - MAX_EVENTS);
            }
            state.debug_event_count = events.length;
            renderState();
            renderEvents();
        }

        function setStatus(text) {
            document.getElementById('status').textContent = text;
        }

        async function bootstrap() {
            const stateResp = await fetch('/debug/state');
            applyStatePatch(await stateResp.json());

            const eventsResp = await fetch('/debug/events?limit=200');
            const eventsData = await eventsResp.json();
            events = Array.isArray(eventsData.events) ? eventsData.events : [];
            state.debug_event_count = Number(eventsData.total || events.length || 0);
            renderState();
            renderEvents();
            setStatus('Connected (snapshot loaded)');
        }

        const socket = io({ transports: ['websocket', 'polling'] });
        socket.on('connect', () => setStatus('Realtime connected'));
        socket.on('disconnect', () => setStatus('Realtime disconnected'));
        socket.on('debug_event', appendEvent);
        socket.on('command_submitted', (data) => {
            const text = ((data || {}).text || '').trim();
            state.command = text || state.command;
            state.command_updated_at = new Date().toISOString();
            state.pending = { ...(state.pending || {}), command: Boolean(text) };
            renderState();
        });
        socket.on('assistant_response', (data) => {
            const text = ((data || {}).text || '').trim();
            state.response = text || state.response;
            state.response_updated_at = new Date().toISOString();
            state.pending = { ...(state.pending || {}), response: Boolean(text) };
            renderState();
        });

        document.getElementById('clearEvents').addEventListener('click', async () => {
            try {
                await fetch('/debug/events/clear', { method: 'POST' });
                events = [];
                state.debug_event_count = 0;
                renderEvents();
                renderState();
                setStatus('Brain log cleared');
            } catch (e) {
                setStatus('Clear failed: ' + e.message);
            }
        });

        setInterval(() => {
            recomputeDerived();
            renderState();
        }, 1000);

        bootstrap().catch((e) => {
            const health = document.getElementById('health');
            health.textContent = 'UNREACHABLE';
            health.className = 'value bad';
            setStatus('Error: ' + e.message);
        });
    </script>
</body>
</html>"""
    return make_response(html, 200, {"Content-Type": "text/html"})


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
