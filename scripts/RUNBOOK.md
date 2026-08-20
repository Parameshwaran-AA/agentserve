# GPU validation runbook

Everything here is free. Take the first option you have access to.

`python -m bench.run` answers whether the policy helps under a modelled cost
function. This answers a harder question: does it help when the cache is vLLM's
own and the latency is wall clock. Nothing in this path is modelled.

## Option A: university HPC cluster

Real multi-GPU nodes, free, and the environment the project describes.

At UB, CCR does not take account requests from students directly. You need a
faculty sponsor to add you to their ColdFront project, which takes them a few
minutes if you already work in a lab or institute.

1. Create the account at https://idm.ccr.buffalo.edu
2. Email your sponsor asking to be added to their project with a UB-HPC
   allocation. One line is enough.
3. Wait for the nightly ColdFront sync at 5pm, then log in.

```bash
git clone <your repo> && cd agentserve
sbatch scripts/ccr_validate.slurm
squeue -u $USER
tail -f agentserve-<jobid>.out
```

Ask for two GPUs, not eight. Small jobs start in minutes and large ones sit in
the queue for hours, and two replicas is enough to show cross-replica routing.

## Option B: Kaggle Notebooks

About 30 GPU hours a week and two separate T4s. Two physically distinct GPUs is
what makes the cross-replica cache problem real rather than simulated, which is
why this is the best free fallback.

1. New notebook, Settings, Accelerator, GPU T4 x2.
2. Open `notebooks/gpu_validation.ipynb` and run the cells.
3. Use the Kaggle variant of cell 3.

Around 20 minutes end to end, most of it downloading the model.

## Option C: Google Colab free tier

One T4, both replicas sharing the card via `--share-gpu`. Session disconnects
are the main annoyance, so keep the run short. Same notebook, Colab variant of
cell 3.

## Option D: paid, only if A through C all fail

RunPod RTX 4090 at roughly $0.35 to $0.70/hr, billed per second. Four hours of
experimentation is a couple of dollars.

```bash
python scripts/gpu_validate.py --replicas 2 --share-gpu \
  --sessions 32 --kv-budget-tokens 40000
```

## Making the experiment measure something

The most common way this produces a null result is a cache that never comes
under pressure. If every session's prefix fits in GPU memory at once, nothing is
evicted, eviction policy is irrelevant, and both arms score about the same. That
is not a refutation. It is a workload with no scarcity, which is not the
situation the project is about.

Check that pressure exists before believing a flat result:

```bash
curl -s localhost:8299/debug/sessions | python -m json.tool
```

Look at `gpu_utilization` and `evictions`. If utilization is low and evictions
are zero, turn one screw at a time:

| Symptom | Change |
|---|---|
| No evictions at all | `--kv-budget-tokens 12000` |
| Utilization under 50% | `--sessions 40` |
| Sessions never overlap | `--concurrency 16` |
| Hit rates equal in both arms | `--prompt-tokens 2000` |

Change one thing per run and write down what you changed.

## Reporting the result

Describe the scale you measured at. This is a small model on one or two GPUs. The
mechanism is validated, the datacenter magnitudes are not.

Good: "Validated on 2x T4 with 4 vLLM replicas of Qwen2.5-0.5B. Cache hit rate
rose from X% to Y% against round-robin plus LRU across 240 replayed agent calls."

Bad: "Built a GPU cluster scheduler that improves throughput 81%."

The first survives follow-up questions. The second does not.

If the numbers come out weaker than the simulation predicted, report the weaker
numbers. "Simulation said 81%, hardware said 22%, and here is my theory about the
gap" is a much better answer than a flattering number you cannot explain.
