# amtavla

CLI assistant with memory and web search.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install llama.cpp

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8

# Download model (Qwen2.5-Coder-7B-Instruct-Q4_0_4_4.gguf)
mkdir -p ~/llama.cpp/models
# Put model in: ~/llama.cpp/models/Qwen2.5-Coder-7B-Instruct-Q4_0_4_4.gguf
```

### 3. Install llama-embed (for embeddings)

```bash
# Just ensure ollama is running with nomic-embed-text
ollama pull nomic-embed-text
```

## Running

### Start Debug API (port 8080)
```bash
python server/debug_api.py
```

### Start Phone UI (port 8081)
```bash
python server/phone_server.py
```

### Run amtavla
```bash
python main.py
```

## Features

- **Debug API**: POST/GET prompts at `http://127.0.0.1:8080/prompts`
- **Phone UI**: Voice interface at `http://127.0.0.1:8081`
- **Memory**: STM (JSON) + LTM tree with embeddings
- **Web Search**: DuckDuckGo with 200-word snippets
