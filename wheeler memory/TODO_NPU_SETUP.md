# TODO: Setup Qwen2.5-1B on Intel NPU via OpenVINO 2026.0

**Status:** Backlog — hardware confirmed, software not yet installed.

## Hardware Status

- [x] Arrow Lake NPU visible (`intel_vpu` loaded)
- [x] `/dev/accel0` present
- [x] User in `render` group
- [ ] NPU driver up-to-date for current kernel

## Software Tasks

- [ ] Install OpenVINO 2026.0 (`pip install openvino==2026.0 openvino-genai==2026.0 optimum-intel`)
- [ ] Verify NPU in `core.available_devices` → `['CPU', 'GPU', 'NPU']`
- [ ] Export Qwen2.5-1.5B-Instruct to OpenVINO IR (INT4, NPU-optimized)
  ```bash
  optimum-cli export openvino \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --weight-format int4 \
    --task text-generation-with-past \
    ./qwen2.5-1b-ov
  ```
- [ ] Run inference on NPU: `LLMPipeline(model_path, device="NPU")`
- [ ] Benchmark: < 1s for 50 tokens
- [ ] Build `speak()` interface for Ralph

## Success Criteria

- [ ] `core.available_devices` includes `NPU`
- [ ] Model loads on `device="NPU"` without error
- [ ] Inference completes with coherent text
- [ ] Latency under 1 second for ~50 tokens
- [ ] CPU and AMD GPU usage near zero during NPU inference
- [ ] `speak()` function works as callable interface

## Architecture

```
Wheeler Memory (CPU / system RAM)
     ↓ compressed context
Qwen2.5-1B (Intel NPU)  ← this task
     ↓ when deep reasoning needed
Qwen3.5-27B (AMD RX 9070 XT / 16GB VRAM)
```

## Notes

- NPU driver may need AUR or Intel GitHub build for CachyOS
- NPU plugin: static shapes only (optimum-cli handles this)
- Fallback: `--weight-format int8` if INT4 hits errors
- Do NOT set ROCR_VISIBLE_DEVICES — conflicts with OpenVINO device detection
