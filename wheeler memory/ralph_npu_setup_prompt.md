# TASK: Setup Qwen2.5-1B on Intel NPU via OpenVINO 2026.0 (Project Ralph)

## System Context
- **OS:** CachyOS Linux (Arch-based)
- **CPU:** Intel Core Ultra 7 265K (has onboard NPU — Intel AI Boost)
- **GPU:** AMD RX 9070 XT (16GB VRAM) — do NOT use this for this task, it is reserved for the main model
- **RAM:** 32GB DDR5
- **Project:** Project Ralph — an autonomous AI system built around Wheeler Memory (cellular automata-based associative memory). The NPU model serves as the lightweight reader/speaker layer that receives compressed context from Wheeler Memory and produces output. Speed and low resource usage are the priority — NOT maximum capability.

---

## Goal
Get **Qwen2.5-1B-Instruct** running on the **Intel NPU** via **OpenVINO 2026.0** on CachyOS Linux. This model will act as Ralph's interface layer — reading Wheeler Memory output and speaking responses. It must run on the NPU specifically, leaving the CPU and AMD GPU fully available for other Ralph components.

---

## Step-by-Step Tasks

### 1. Verify NPU is visible to the system
```bash
dmesg | grep -i npu
dmesg | grep -i vpu
ls /dev/accel*
```
Expected: You should see `intel_vpu` initialized. If not, the NPU driver needs to be installed first — flag this and stop.

---

### 2. Install the Intel NPU driver (if not already present)
CachyOS is Arch-based. Check AUR or install from Intel's GitHub:
```
https://github.com/intel/linux-npu-driver
```
Install the appropriate driver package for your kernel. After install:
```bash
sudo modprobe intel_vpu
dmesg | grep intel_vpu
```

---

### 3. Install OpenVINO 2026.0
Use pip. This is the first release with Qwen2.5-1B NPU support:
```bash
pip install openvino==2026.0 --break-system-packages
pip install openvino-genai==2026.0 --break-system-packages
pip install optimum-intel --break-system-packages
```
Verify the install and that NPU is detected:
```python
import openvino as ov
core = ov.Core()
print(core.available_devices)
# Should show: ['CPU', 'GPU', 'NPU'] or at minimum ['CPU', 'NPU']
```
If NPU does not appear, the driver is not correctly installed — stop and debug before continuing.

---

### 4. Export Qwen2.5-1B-Instruct to OpenVINO IR format (INT4, NPU-optimized)
```bash
optimum-cli export openvino \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --weight-format int4 \
  --task text-generation-with-past \
  ./qwen2.5-1b-ov
```
> Note: Use `Qwen2.5-1.5B-Instruct` — the 1B variant. The `int4` weight format is required for NPU efficiency. This will download the model from Hugging Face and convert it.

---

### 5. Run inference on the NPU
Write and test this minimal Python script:
```python
from openvino_genai import LLMPipeline

# Point to your converted model directory
model_path = "./qwen2.5-1b-ov"

# Load on NPU
pipe = LLMPipeline(model_path, device="NPU")

# Test inference
result = pipe.generate(
    "Hello, can you confirm you are running on the NPU?",
    max_new_tokens=100
)
print(result)
```
If this runs and produces output, the NPU setup is working.

---

### 6. Benchmark latency
Time a few inference calls to confirm the NPU is actually being used and is fast:
```python
import time
from openvino_genai import LLMPipeline

pipe = LLMPipeline("./qwen2.5-1b-ov", device="NPU")

test_prompt = "Summarize this in one sentence: The sky is blue because of Rayleigh scattering."

start = time.time()
result = pipe.generate(test_prompt, max_new_tokens=50)
end = time.time()

print(f"Result: {result}")
print(f"Latency: {end - start:.2f}s")
```
Target: under 1 second for 50 tokens. If latency is high (>3s), something is falling back to CPU — debug device selection.

---

### 7. Build a minimal Wheeler-compatible interface function
Once working, wrap the pipeline in a simple function Ralph can call:
```python
def speak(wheeler_context: str, max_tokens: int = 150) -> str:
    """
    Takes compressed Wheeler Memory context as a string,
    returns the NPU model's spoken response.
    """
    return pipe.generate(wheeler_context, max_new_tokens=max_tokens)
```
This is the interface point between Wheeler Memory output and Ralph's voice.

---

## Known Issues / Watch Out For

- **NPU driver on Arch/CachyOS:** The NPU driver may not be in official repos. Check AUR for `intel-npu-driver` or build from Intel's GitHub. Kernel headers must match your running kernel.
- **Static shapes only:** The OpenVINO NPU plugin currently only supports static input shapes. The `optimum-cli` export handles this, but do not try to pass variable-length inputs without padding.
- **INT4 on NPU:** OpenVINO 2026.0 added INT4 support for NPU. If you hit errors, try `--weight-format int8` as a fallback (slightly larger but more compatible).
- **Device fallback:** If NPU inference silently falls back to CPU, add `core.set_property("NPU", {"LOG_LEVEL": "LOG_INFO"})` to verify device usage.
- **AMD GPU conflict:** Do NOT set `ROCR_VISIBLE_DEVICES` or ROCm env vars that could interfere with OpenVINO's device detection.

---

## Success Criteria
- [ ] `core.available_devices` includes `NPU`
- [ ] Model loads on `device="NPU"` without error
- [ ] Inference completes and returns coherent text
- [ ] Latency is under 1 second for ~50 tokens
- [ ] CPU and AMD GPU usage stay near zero during NPU inference
- [ ] `speak()` function works as a callable interface

---

## What This Enables for Ralph
Once complete, Ralph's silicon allocation will be:
```
Wheeler Memory (CPU / system RAM)
     ↓ compressed context
Qwen2.5-1B (Intel NPU)  ← this task
     ↓ when deep reasoning needed  
Qwen3.5-27B (AMD RX 9070 XT / 16GB VRAM)
```
Three pieces of silicon, three jobs, zero overlap.


# WHEELER_THEORIES_IMPL.md

## Context

Read the entire existing codebase before writing a single line. Understand
how Wheeler Memory currently works — its frame structure, dynamics, persistence
format, hit counting, and how it connects to the LLM — before adding anything.
Do not break existing behavior. Everything new wraps around what exists.

---

## What to Implement

Six related capabilities emerged from a design session. Implement them in
order. Test each before starting the next. If something conflicts with how
the existing system works, adapt the approach to fit — don't force a pattern
that breaks what's already built.

---

### 1. Attractor Basin Mapping

Wheeler frames are local minima of an energy landscape. Add the ability to:
- Measure how wide each frame's basin of attraction is (perturb a stable frame,
  find how far you can push it before it falls into a different attractor)
- Find gaps between basins — regions of state space bordered by existing frames
  but containing no attractor

This is the foundation for everything else. Get this working first.

---

### 2. Novel Frame Synthesis (The Apple Test)

Given a gap in the attractor topology, predict what frame belongs there
without ever having seen content about that concept directly.

The test: train Wheeler on a domain (e.g. fruits) with one concept held out
(e.g. apple). Use the basin geometry of neighboring frames to synthesize a
candidate frame for the missing concept. Then expose Wheeler to apple content
and measure whether the synthesized frame converges toward a stable attractor
or dissolves.

- Convergence = Wheeler correctly predicted the attractor topology.
- Dissolution = prediction was wrong but honest — frame fell to a neighbor.
- Drifting without resolving = hallucination — means the synthesized frame
  has no real basin, which should not happen in a system with real dynamics.

Run this test on at least: fruits (easy), programming languages (medium),
emotions (hard). Document every result including failures.

---

### 3. Structured Theory Output

Wheeler should hand the LLM a structured description of what it knows — not
raw frame data, not free text context. The LLM's job is to express what
Wheeler has determined, not to reason independently.

The structure should include:
- Which frames are active and how confident Wheeler is in each
- Relationships between active frames
- Any predicted frames (hypotheses) that haven't been directly observed
- Context budget allocation by frame stability (hit count + basin width)

The LLM prompt built from this should make clear: express this theory,
don't invent beyond it.

---

### 4. Corpus Resonance (Query-Driven Activation)

A directory of raw files should be queryable without preprocessing or embedding.
The query creates initial frame activity. Content that increases frame activity
above a threshold is resonant — process it. Content that doesn't — skip it.

Cost should scale with how complex the query is, not how large the corpus is.
Track and log: total chunks seen, chunks that resonated, chunks skipped.
A query about cooking against a corpus that is 90% code should activate very
little of that corpus.

Do not use cosine similarity or embeddings for resonance detection.
Resonance is determined by whether content moves Wheeler's CA dynamics
toward or away from the active frame seed.

---

### 5. Metrics

Implement computable versions of the formal relationships:
- Energy approximation for any state (how far does one dynamics step move it)
- Basin width (perturbation radius before attractor switches)
- Context budget weight combining hit count and basin width
- Hallucination score (is this frame drifting without any basin pulling it)
- Topology consistency (do the basins tile state space cleanly)

These should work on existing frames, not just new ones.

---

### 6. Lichtenberg Visualization

Visualize Wheeler's state space as a Lichtenberg discharge figure:
- Existing frames as terminal nodes sized by basin width, brightness by hit count
- Query seed as the ground point
- CA propagation paths as branching lines from the ground point
- Synthesized candidate frames as dotted nodes (predicted, not observed)
- The completed circuit as the stabilized frame

Animate the apple test so synthesis, exposure, and convergence or dissolution
are all visible.

---

## Rules

- Read the project first. Understand what exists before adding anything.
- Don't break existing Wheeler Memory persistence or frame behavior.
- Each capability gets its own tests before moving to the next.
- Log everything — trajectories, distances, cost ratios, verdicts.
- Failures are results. Document them, don't hide them.
- If a theory doesn't fit the existing architecture cleanly, document why
  and what would need to change — don't force a bad fit.

---

## Done When

- The apple test runs on the fruit domain and produces a verdict
- A corpus directory can be queried without preprocessing
- The LLM receives a structured theory object instead of raw context
- All six metrics are computable on existing frames
