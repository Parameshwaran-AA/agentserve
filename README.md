# AgentServe

An LLM serving gateway that schedules agent sessions instead of individual requests.

Serving engines like vLLM are built around one request at a time. Agents do not send
one request. They send thirty, each containing the previous one word for word, with
tool calls in between that take anywhere from 300ms to two minutes. That mismatch
throws away KV cache right before it would have been reused. AgentServe makes the
session the scheduling unit and keeps the cache where the next call will land.

```
python -m bench.run --sessions 300 --seeds 8
```

| | round-robin + LRU | AgentServe |
|---|---:|---:|
| GPU cache hit rate | 33.3% | **71.1%** |
| Prompt tokens reused | 31.2% | **94.7%** |
| GPU compute time | 6,457s | **1,227s** |
| p95 job completion | 663.5s | **498.6s** |

Simulated, not yet measured on GPUs. See [Honest scope](#honest-scope).

---

## The problem

A coding agent given one task ("fix the failing test") makes around 25 calls to the
model. Call 4's prompt is call 3's prompt plus the new tool output. By the end of a
task the prompt is 40,000 tokens and roughly 95% of it is byte-identical to the
previous call.

When a GPU processes a prompt it saves intermediate results per token, the KV cache.
If that cache survives, the next call only prefills the 500 new tokens: about 30ms
instead of 2 seconds. If it does not survive, the GPU redoes all 40,000 tokens.

Two things destroy it, and they compound.

**Round-robin routing.** A Kubernetes Service has no idea KV cache exists. Call 4
lands on a replica that never saw the conversation, a guaranteed miss even though the
cache is sitting warm on another node. With four replicas you get lucky about a
quarter of the time, and you have built four copies of the same cache.

**LRU eviction.** While the agent runs `pytest` for 45 seconds it has nothing in
flight, so it looks idle and gets evicted. LRU cannot tell a paused agent from a
finished one. Given a session that paused 40 seconds ago and returns in 5, and a
session touched 2 seconds ago whose user closed their laptop, LRU evicts the first.
Recency tells you about the past. What you need is a guess about the future.

Across a 25-call task that is roughly a minute of wasted GPU time per agent. At a
thousand concurrent agents, redundant prefill is most of the hardware bill.

## The approach

A gateway between the agent and the vLLM replicas. Clients keep their existing
OpenAI SDK and add one header.

```
        agent  --  X-Session-Id: task-42
          |
    +-----v-----+   which replica?
    | AgentServe|   keep, demote, or drop the cache?
    +--+--+--+--+
       |  |  |
      GPU GPU GPU     HBM: hot prefixes
        \ | /
      host DRAM       paused sessions parked here
```

**Session affinity.** A `session -> replica` map sends follow-up calls back to the
replica holding the prefix. Affinity is a preference, not a pin: past a configured
in-flight depth the call goes to the least loaded replica instead, otherwise a hot
session serializes behind itself while the cluster idles. A load-driven detour does
not re-home the session, because a momentary queue spike should not cost it a warm
40,000-token prefix permanently.

**Three-way cache retention.** When a response is sent, estimate when the session
returns. Under ~2s it stays pinned in GPU memory. Between that and ~5 minutes it is
demoted to host DRAM, which is roughly 16x cheaper to restore than recomputing.
Beyond that it is dropped. When memory pressure forces an eviction, the victim is
chosen by predicted return time rather than last touch, so a session due back in
300ms outranks one touched more recently but away for 40 seconds.

**Learned tool durations.** Nothing in the request says how long the agent will be
gone, so it is learned: an EWMA per tool name, falling back to a per-session average
and then a global prior. After a few minutes the gateway knows `grep` returns in
400ms and `run_tests` in 44s. EWMA rather than a mean because tool timings drift
within a session as test suites grow. No workflow graph, no offline profiling, no
change to the agent beyond one header.

## Results

Both arms replay identical traces through identical hardware assumptions. Only the
router and the cache policy differ, so the delta is attributable to the scheduler.
Averaged over 8 seeds: 300 sessions, 7,360 calls, ~18,000 mean prompt tokens, 4
replicas with a 150,000-token KV budget each, ~81% baseline utilization.

| Metric | round-robin + LRU | AgentServe | Delta |
|---|---:|---:|---:|
| GPU cache hit rate | 33.3% | 71.1% | +37.8 pp |
| Hit rate including DRAM tier | 33.3% | 95.9% | +62.6 pp |
| Prompt tokens reused | 31.2% | 94.7% | +63.5 pp |
| Tokens prefilled cold | 89.3M | 6.9M | -92.3% |
| GPU compute time | 6,457s | 1,227s | -81.0% |
| p95 time-to-first-token | 26,844ms | 284ms | -98.9% |
| p95 job completion time | 663.5s | 498.6s | -24.9% |

Hit-rate standard deviation across seeds: 0.003.

Which of these to believe:

- **The 81% compute saving is the real result.** It follows from the token reuse rate
  and does not depend on load level. This is the one that turns into money.
- **The 98.9% TTFT figure is a side effect, not a separate win.** At 81% utilization
  the baseline wastes enough GPU time on redundant prefill to tip into queueing
  collapse. On a lightly loaded cluster the same comparison gives about -88%.
- **The 24.9% job completion improvement is the conservative number and the one a
  user would feel.** Job completion includes the agent's own tool execution, about
  170s of test runs and builds per session that no scheduler can speed up.

## Honest scope

**The numbers above are simulated.** The benchmark backend is a deterministic cost
model, with every hardware assumption in `HardwareProfile` and nowhere else. The
scheduler, policy, predictor, router and metrics are real code; only the GPU timing
is modelled.

**The serving path is real and tested.** `AGENTSERVE_BACKEND=vllm` selects
`backends/vllm.py`, driven end to end in CI by `scripts/integration_check.py` against
mock workers speaking vLLM's actual wire format: prefix-cache usage fields, SSE
streaming, retry, failover.

**GPU validation is written and free to run.** `scripts/gpu_validate.py` starts real
vLLM workers, replays the same workload through the same gateway binary twice, and
reports the measured difference. It fits a free Kaggle or Colab T4. See
[`scripts/RUNBOOK.md`](scripts/RUNBOOK.md).

**Known limits.** The baseline is round-robin plus LRU, which is what most deployments
run, but not a tuned workflow-graph scheduler. Traces are synthetic, matched to
published workload shapes rather than captured from production. No multi-tenant
fairness, priority or preemption. The predictor keys only on tool name, not repository
size or machine load.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                                     # 130 tests
python -m bench.run --sessions 300 --seeds 8  # reproduces the table
python scripts/integration_check.py           # end to end against mock workers
```

Serve it:

```bash
uvicorn agentserve.gateway:app --port 8000
curl -s localhost:8000/health
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'X-Session-Id: task-42' \
  -H 'X-Tool-Name: run_tests' \
  -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":256}'
```

The response carries an `agentserve` block with the replica served, cache tier hit,
tokens reused versus recomputed, and the predicted gap. Add `"stream": true` for SSE.
Streaming is not cosmetic: TTFT is the metric this optimizes, and a client cannot
observe it through a non-streaming response.

Full stack with Redis, Prometheus and Grafana:

```bash
docker compose up --build    # gateway :8000, prometheus :9090, grafana :3000
```

## Validating on real GPUs

The simulated benchmark answers whether the policy helps under a modelled cost
function. This answers the harder question, where the cache is vLLM's own and the
latency is wall clock:

```bash
python scripts/gpu_validate.py --dry-run --replicas 2    # check the plan
python scripts/gpu_validate.py --replicas 2              # 2 GPUs
python scripts/gpu_validate.py --replicas 2 --share-gpu  # 1 GPU
```

It starts the workers, runs both arms through the same binary, cools the caches
between them so the control is not measured cold while the treatment runs warm, and
writes `gpu_validation.json`.

A 0.5B model is deliberate. Routing and eviction are what is being tested, and a small
model exercises vLLM's prefix cache with the same `cached_tokens` accounting a 70B
does. One free T4 tests whether the mechanism works. It does not test datacenter
throughput and the script prints no such claim.

The most common null result is a cache that never comes under pressure. Give every
session room and nothing is evicted, so eviction policy is irrelevant and both arms
tie. Check `gpu_utilization` and `evictions` on `/debug/sessions` before concluding
anything from a flat result. The runbook lists what to turn.

## Configuration

Every knob is an environment variable read by `agentserve/settings.py`, and the Helm
ConfigMap sets exactly those keys. `tests/test_deploy_contract.py` fails the build if
the two drift apart, because a ConfigMap key nothing reads is worse than no ConfigMap
at all: the cluster looks configured and is not.

| Variable | Default | Purpose |
|---|---|---|
| `AGENTSERVE_BACKEND` | `simulated` | `simulated` or `vllm` |
| `AGENTSERVE_VLLM_ENDPOINTS` | | Comma-separated worker URLs, required for `vllm` |
| `AGENTSERVE_REDIS_URL` | | Shared session map, required for more than one gateway pod |
| `AGENTSERVE_ROUTER` | `affinity` | `affinity` or `round-robin`, lets the same binary be the control arm |
| `AGENTSERVE_POLICY` | `adaptive` | `adaptive` or `lru` |
| `AGENTSERVE_API_KEY` | | Bearer token, auth off when unset |
| `AGENTSERVE_PIN_THRESHOLD_MS` | `2000` | Below this predicted gap, pin in HBM |
| `AGENTSERVE_OFFLOAD_THRESHOLD_MS` | `300000` | Below this, offload to DRAM instead of dropping |
| `AGENTSERVE_AFFINITY_QUEUE_LIMIT` | `6` | In-flight depth past which affinity yields to load |
| `AGENTSERVE_GPU_CACHE_TOKENS` | `150000` | Per-replica KV budget |

Bad values fail at startup naming the variable, not on the thousandth request.

## Scaling the gateway

The routing table is state. With more than one gateway pod and no shared store, pod A
routes session 42 to replica 3, pod B has never heard of session 42 and routes it to
replica 1, and affinity degrades to round-robin **silently**. Traffic still flows,
every request succeeds, and the hit rate quietly collapses with no error anywhere.

- One pod: the in-memory store is correct. It is a bounded LRU with TTL, not an
  unbounded dict, so a long-running process does not leak.
- More than one: set `AGENTSERVE_REDIS_URL`. The Helm chart **refuses to render**
  `gateway.replicaCount > 1` without `redis.enabled=true`, because a failure that
  never announces itself has to be caught at deploy time.

Redis is a cache, not a source of truth. Every entry is disposable, so a stale or
missing mapping costs one redundant prefill and the session re-homes on its next call.
If Redis is unreachable the gateway degrades to a cache miss rather than raising: it
sits on the hot path for every agent call and must not become a new way for inference
to fail. No quorum, no leader election.

## Deployment

```bash
helm install agentserve deploy/helm \
  --set vllm.replicaCount=8 \
  --set vllm.model=mistralai/Mistral-7B-Instruct-v0.3 \
  --set gateway.replicaCount=3 \
  --set redis.enabled=true \
  --set redis.url=redis://agentserve-redis:6379/0
```

vLLM runs as a StatefulSet rather than a Deployment because session affinity needs
stable per-replica DNS: a Deployment hands out new pod names and every mapping goes
stale at once. The chart sets `--enable-prefix-caching` (the cache the router steers
toward) and `--swap-space` (what backs the DRAM tier). vLLM will swap; it just will
not decide when on agent-aware grounds, which is what the policy is for.

Liveness hits `/health`, which touches no dependency, so a Redis blip does not get a
healthy pod restarted. Readiness hits `/ready`, which does check the workers, so a pod
that cannot reach them leaves the Service.

## Layout

```
agentserve/
  config.py       hardware profile and policy thresholds, all assumptions here
  models.py       Turn, SessionTrace, CacheEntry, RequestResult, RunStats
  predictor.py    EWMA tool-duration estimation
  policy.py       LruPolicy (baseline) vs AdaptiveTtlPolicy
  replica.py      two-tier KV cache, capacity enforcement, in-flight slots
  router.py       RoundRobinRouter (baseline) vs SessionAffinityRouter
  state.py        session map: bounded in-memory or shared Redis
  settings.py     environment to config, validated at startup
  tokenizer.py    real token counting via tiktoken
  metrics.py      Prometheus counters, histograms, gauges
  gateway.py      FastAPI service, streaming and auth
  backends/       simulated cost model | real vLLM adapter

bench/
  traces.py       synthetic agent traces
  replay.py       discrete-event simulation, A/B harness
  run.py          CLI benchmark with multi-seed variance
  live_replay.py  HTTP replay against a live gateway, nothing modelled

scripts/
  gpu_validate.py       launches vLLM workers, runs the A/B on real GPUs
  ccr_validate.slurm    Slurm batch job for an HPC cluster
  RUNBOOK.md            free ways to get a GPU, and how to read the result
  integration_check.py  end-to-end drive of the real vLLM path
  mock_vllm.py          worker stub speaking vLLM's wire format

notebooks/gpu_validation.ipynb    Kaggle / Colab, free tier
deploy/helm/                      gateway, vLLM StatefulSet, Redis, ServiceMonitor
deploy/grafana/                   hit rate, tier mix, cache utilization, tool gaps
tests/                            130 tests across every module
```

## Prior work

The problem is documented. SAGA measured agents spending 38% of total time
regenerating discarded KV cache, with request-level scheduling inflating end-to-end
latency 3-8x. Tokencake reports large end-to-end reductions on multi-agent benchmarks
by scheduling around tool-call gaps.

What is combined here that those do not: the TTL decision is driven by an online
duration estimate rather than a supplied workflow graph, the DRAM offload decision is
made jointly with routing instead of independently, and affinity yields to load so a
hot session cannot serialize behind itself. Whether that combination beats a
well-tuned workflow-graph scheduler is an open question this repo does not answer.

## License

MIT
