"""Stand-in for a vLLM worker, used by scripts/integration_check.py and CI.

Speaks the same OpenAI-compatible wire format including
prompt_tokens_details.cached_tokens and SSE streaming, and keeps a real prefix
cache so growing prompts produce genuine hits. Lets the production code path be
tested without a GPU. Does not simulate GPU timing.
"""
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
CACHE = {}   # worker-local prefix cache, keyed by prompt prefix hash

@app.get("/health")
def health(): return {"status": "ok"}

def account(text):
    prompt = max(1, len(text)//4)
    cached = 0
    for k, v in list(CACHE.items()):
        if text.startswith(k):
            cached = max(cached, v)
    CACHE[text] = prompt
    return prompt, cached

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    text = "".join(m.get("content","") for m in body["messages"])
    prompt, cached = account(text)
    usage = {"prompt_tokens": prompt, "completion_tokens": 5,
             "prompt_tokens_details": {"cached_tokens": cached}}
    if body.get("stream"):
        def gen():
            for w in ["fix", "ed ", "the ", "bug"]:
                yield f'data: {json.dumps({"choices":[{"delta":{"content":w}}]})}\n\n'
                time.sleep(0.01)
            yield f'data: {json.dumps({"choices":[],"usage":usage})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices":[{"message":{"role":"assistant","content":"fixed the bug"}}], "usage": usage}
