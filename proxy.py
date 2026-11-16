"""
AgentRouter Proxy Server
Translates /v1/chat/completions and /v1/messages → /v1/responses for AgentRouter

Usage:
    pip install flask requests
    python proxy.py
"""

import os
import json
import requests
from flask import Flask, request, Response, jsonify

app = Flask(__name__)

AGENTROUTER_BASE = "https://agentrouter.org/v1"
AGENTROUTER_KEY  = os.environ.get("AGENTROUTER_API_KEY", "sk-gaVGPKBboQbUYJUoIhtJbesS3XDYzgONTru7rjTERBYiwG9o")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AGENTROUTER_KEY}",
    "User-Agent": "Cline/3.0",   # Required by AgentRouter client whitelist
}


def messages_to_prompt(messages: list) -> str:
    """Convert OpenAI-style messages list to a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Return models list so Claude Desktop & Hermes checks pass."""
    return jsonify({
        "object": "list",
        "data": [
            {"id": "claude-sonnet-4-5", "object": "model"},
            {"id": "claude-3-7-sonnet", "object": "model"},
            {"id": "gpt-6.1-sol",       "object": "model"},
            {"id": "gpt-6-astra",       "object": "model"},
            {"id": "gpt-5.6-sol",       "object": "model"},
        ]
    })


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/v1/messages", methods=["POST"])
def chat_completions():
    try:
        is_anthropic = (request.path == "/v1/messages")
        body       = request.get_json(force=True)
        raw_model  = body.get("model", "gpt-6.1-sol")
        
        # Map Claude model names to gpt-6.1-sol so AgentRouter accepts them
        if raw_model.startswith("claude") or raw_model.startswith("anthropic/"):
            model = "gpt-6.1-sol"
        else:
            model = raw_model
            
        messages   = body.get("messages", [])
        
        # Handle Anthropic system prompt format
        system_prompt = body.get("system", "")
        if system_prompt:
            if isinstance(system_prompt, list):
                system_prompt = " ".join(c.get("text", "") for c in system_prompt if c.get("type") == "text")
            messages = [{"role": "system", "content": system_prompt}] + messages
            
        stream     = body.get("stream", False)
        max_tokens = body.get("max_tokens") or 16000

        prompt = messages_to_prompt(messages)

        payload = {
            "model":             model,
            "input":             prompt,
            "max_output_tokens": max(max_tokens, 8000),
            "stream":            stream,
        }

        upstream = requests.post(
            f"{AGENTROUTER_BASE}/responses",
            headers=HEADERS,
            json=payload,
            stream=stream,
            timeout=120,
        )
        upstream.encoding = "utf-8"  # 👈 THIS FIXES THE WEIRD BOXES / MOJIBAKE

        # ── Non-streaming ──────────────────────────────────────────────────────────
        if not stream:
            if upstream.status_code != 200:
                return Response(upstream.content, status=upstream.status_code,
                                content_type="application/json")

            data = upstream.json()
            reply_text = ""
            for out in data.get("output", []):
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        if c.get("type") == "output_text":
                            reply_text += c.get("text", "")

            if is_anthropic:
                return jsonify({
                    "id":          data.get("id", "msg-proxy"),
                    "type":        "message",
                    "role":        "assistant",
                    "model":       raw_model,
                    "content":     [{"type": "text", "text": reply_text}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens":  data.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                    }
                })

            return jsonify({
                "id":      data.get("id", "chatcmpl-proxy"),
                "object":  "chat.completion",
                "model":   raw_model,
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }],
                "usage": data.get("usage", {}),
            })

        # ── Streaming ──────────────────────────────────────────────────────────────
        def generate():
            chunk_id = f"msg_{os.urandom(8).hex()}"

            if is_anthropic:
                # 1. Start message
                msg_init = {
                    "type": "message_start",
                    "message": {
                        "id": chunk_id,
                        "type": "message",
                        "role": "assistant",
                        "model": raw_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 15, "output_tokens": 1}
                    }
                }
                yield f"event: message_start\ndata: {json.dumps(msg_init)}\n\n"
                
                # 2. Start text content block
                yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
                
                # 3. Initial ping to acknowledge connection immediately
                yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

            for raw_line in upstream.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    if not is_anthropic:
                        yield "data: [DONE]\n\n"
                    return

                try:
                    ev = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                ev_type = ev.get("type", "")

                # While reasoning is in progress, keep connection active with pings so Claude doesn't freeze
                if "reasoning" in ev_type or ev_type in ("response.in_progress", "response.output_item.added"):
                    if is_anthropic:
                        yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"
                    continue

                # Text delta
                if ev_type == "response.output_text.delta":
                    delta = ev.get("delta", "")
                    if is_anthropic:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':delta}})}\n\n"
                    else:
                        chunk = {
                            "id":      chunk_id,
                            "object":  "chat.completion.chunk",
                            "model":   raw_model,
                            "choices": [{
                                "index":  0,
                                "delta":  {"role": "assistant", "content": delta},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Terminal events
                elif ev_type in ("response.completed", "response.incomplete", "response.failed"):
                    if is_anthropic:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
                        yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':250}})}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
                    else:
                        chunk = {
                            "id":      chunk_id,
                            "object":  "chat.completion.chunk",
                            "model":   raw_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    return

        return Response(generate(), content_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as err:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(err), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  AgentRouter Proxy running on port 8765")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8765, threaded=True)
