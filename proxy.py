import os, json, requests
from flask import Flask, request, Response, jsonify

app = Flask(__name__)
AGENTROUTER_BASE = "https://agentrouter.org/v1"
AGENTROUTER_KEY  = os.environ.get("AGENTROUTER_API_KEY", "")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AGENTROUTER_KEY}",
    "User-Agent": "Cline/3.0",
}

def messages_to_prompt(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text","") for c in content if c.get("type")=="text")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({"object":"list","data":[
        {"id":"gpt-6.1-sol","object":"model"},
        {"id":"gpt-6-astra","object":"model"},
        {"id":"gpt-5.6-sol","object":"model"},
    ]})

def handle_request(body):
    model      = body.get("model", "gpt-6.1-sol")
    messages   = body.get("messages") or []
    stream     = body.get("stream", False)
    max_tokens = max(body.get("max_tokens", 16000), 8000)
    prompt     = messages_to_prompt(messages)

    payload = {"model": model, "input": prompt,
               "max_output_tokens": max_tokens, "stream": stream}

    return requests.post(f"{AGENTROUTER_BASE}/responses",
                         headers=HEADERS, json=payload,
                         stream=stream, timeout=120), model, stream

@app.route("/v1/chat/completions", methods=["POST"])   # OpenAI format
@app.route("/v1/messages", methods=["POST"])           # Anthropic format
def completions():
    body = request.get_json(force=True)

    # Normalize Anthropic format → OpenAI messages
    if request.path == "/v1/messages":
        sys = body.get("system","")
        msgs = []
        if sys: msgs.append({"role":"system","content":sys})
        msgs += body.get("messages", [])
        body["messages"] = msgs

    upstream, model, stream = handle_request(body)

    if not stream:
        if upstream.status_code != 200:
            return Response(upstream.content, status=upstream.status_code,
                            content_type="application/json")
        data = upstream.json()
        reply = ""
        for out in data.get("output", []):
            if out.get("type") == "message":
                for c in out.get("content", []):
                    if c.get("type") == "output_text":
                        reply += c.get("text","")

        # Return in correct format
        if request.path == "/v1/messages":
            return jsonify({"id":data.get("id","msg-proxy"),"type":"message",
                "role":"assistant","model":model,
                "content":[{"type":"text","text":reply}],
                "stop_reason":"end_turn","usage":data.get("usage",{})})
        return jsonify({"id":data.get("id","chatcmpl-proxy"),"object":"chat.completion",
            "model":model,"choices":[{"index":0,
            "message":{"role":"assistant","content":reply},"finish_reason":"stop"}],
            "usage":data.get("usage",{})})

    def generate():
        for raw in upstream.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"): continue
            ds = raw[5:].strip()
            if ds == "[DONE]": yield "data: [DONE]\n\n"; return
            try: ev = json.loads(ds)
            except: continue
            t = ev.get("type","")
            if t == "response.output_text.delta":
                delta = ev.get("delta","")
                chunk = {"id":"proxy","object":"chat.completion.chunk","model":model,
                    "choices":[{"index":0,"delta":{"role":"assistant","content":delta},
                    "finish_reason":None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            elif t in ("response.completed","response.incomplete","response.failed"):
                chunk = {"id":"proxy","object":"chat.completion.chunk","model":model,
                    "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
                yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"; return

    return Response(generate(), content_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Proxy running on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
