import os
import time
from datetime import datetime, timezone
from flask import Flask, send_file, jsonify, make_response, request

app = Flask(__name__)

UI_FILE = os.path.join(os.path.dirname(__file__), "phone_ui.html")
command_store = None
response_store = None
command_updated_at = None
response_updated_at = None
server_started_at = time.time()


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
    }


@app.route("/")
def index():
    return send_file(UI_FILE)


@app.route("/command", methods=["POST"])
def set_command():
    global command_store, command_updated_at
    data = request.get_json()
    if data and "text" in data:
        command_store = data["text"]
        command_updated_at = time.time()
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
    global response_store, response_updated_at
    data = request.get_json()
    if data and "text" in data:
        response_store = data["text"]
        response_updated_at = time.time()
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
    </style>
</head>
<body>
    <h1>amtavla Debug</h1>
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
    </div>
    <div class="card">
        <div class="label">Status</div>
        <div class="value" id="status">Waiting...</div>
    </div>
    <script>
        const STALE_SECONDS = 30;

        function setValue(id, value, emptyText='(empty)') {
            const el = document.getElementById(id);
            const hasValue = value !== null && value !== undefined && value !== '';
            el.textContent = hasValue ? value : emptyText;
            el.className = hasValue ? 'value' : 'value empty';
        }

        async function poll() {
            const stateResp = await fetch('/debug/state');
            const data = await stateResp.json();

            setValue('command', data.command);
            setValue('response', data.response);
            setValue('commandAge', data.command_age_seconds, 'n/a');
            setValue('responseAge', data.response_age_seconds, 'n/a');
            setValue('uptime', data.uptime_seconds, 'n/a');
            setValue('commandUpdated', data.command_updated_at, 'n/a');
            setValue('responseUpdated', data.response_updated_at, 'n/a');

            const commandStale = data.pending.command && data.command_age_seconds !== null && data.command_age_seconds > STALE_SECONDS;
            const responseStale = data.pending.response && data.response_age_seconds !== null && data.response_age_seconds > STALE_SECONDS;
            const health = document.getElementById('health');
            if (commandStale || responseStale) {
                health.textContent = 'STALE';
                health.className = 'value warn';
            } else {
                health.textContent = 'OK';
                health.className = 'value ok';
            }

            document.getElementById('status').textContent = 'Last update: ' + new Date().toLocaleTimeString();
        }

        async function loop() {
            try {
                await poll();
            } catch (e) {
                const health = document.getElementById('health');
                health.textContent = 'UNREACHABLE';
                health.className = 'value bad';
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }

        loop();
        setInterval(loop, 1000);
    </script>
</body>
</html>"""
    return make_response(html, 200, {"Content-Type": "text/html"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8081, debug=False)
