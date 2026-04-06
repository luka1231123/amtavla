import os
from flask import Flask, send_file, jsonify, make_response, request

app = Flask(__name__)

UI_FILE = os.path.join(os.path.dirname(__file__), "phone_ui.html")
command_store = None
response_store = None


@app.route("/")
def index():
    return send_file(UI_FILE)


@app.route("/command", methods=["POST"])
def set_command():
    global command_store
    data = request.get_json()
    if data and "text" in data:
        command_store = data["text"]
        return jsonify({"status": "ok"})
    return jsonify({"error": "No text provided"}), 400


@app.route("/command", methods=["GET"])
def get_command():
    global command_store
    text = command_store
    return jsonify({"command": text})


@app.route("/command/ack", methods=["POST"])
def ack_command():
    global command_store
    command_store = None
    return jsonify({"status": "ok"})


@app.route("/response", methods=["POST"])
def set_response():
    global response_store
    data = request.get_json()
    if data and "text" in data:
        response_store = data["text"]
        return jsonify({"status": "ok"})
    return jsonify({"error": "No text provided"}), 400


@app.route("/response", methods=["GET"])
def get_response():
    global response_store
    text = response_store
    return jsonify({"response": text})


@app.route("/response/ack", methods=["POST"])
def ack_response():
    global response_store
    response_store = None
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
        .ok { color: #4caf50; }
        .empty { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>amtavla Debug</h1>
    <div class="card">
        <div class="label">Command Store</div>
        <div class="value" id="command"></div>
    </div>
    <div class="card">
        <div class="label">Response Store</div>
        <div class="value" id="response"></div>
    </div>
    <div class="card">
        <div class="label">Status</div>
        <div class="value" id="status">Waiting...</div>
    </div>
    <script>
        async function poll() {
            const cmdResp = await fetch('/command');
            const cmdData = await cmdResp.json();
            const cmdEl = document.getElementById('command');
            cmdEl.textContent = cmdData.command || '(empty)';
            cmdEl.className = cmdData.command ? 'value ok' : 'value empty';

            const respResp = await fetch('/response');
            const respData = await respResp.json();
            const respEl = document.getElementById('response');
            respEl.textContent = respData.response || '(empty)';
            respEl.className = respData.response ? 'value ok' : 'value empty';

            document.getElementById('status').textContent = 'Last update: ' + new Date().toLocaleTimeString();
        }
        poll();
        setInterval(poll, 1000);
    </script>
</body>
</html>"""
    return make_response(html, 200, {"Content-Type": "text/html"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8081, debug=False)
