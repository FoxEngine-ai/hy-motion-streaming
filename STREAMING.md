# Streaming, motion continuation, and the REST API

> **Scope.** This document covers everything specific to the FoxEngine fork:
> the streaming gradio variant (`gradio_app_streaming.py`), the motion
> continuation feature, and the headless REST API server
> (`streaming_server.py`). For the original (non-streaming) gradio app,
> see Tencent's instructions in [README.md](README.md).

---

## Streaming gradio interface

The streaming variant generates motion frame-by-frame and pushes each
frame to the browser as it lands, rather than holding everything until
the full clip is ready. Total generation time is unchanged; what changes
is that the user sees progress instead of staring at a spinner.

### How it works

The standard ODE solver computes the entire motion trajectory in one
pass. The streaming path swaps that for a generator that yields after
each integration step, decodes the intermediate latent into a partial
motion, and ships it through a queue to the gradio HTML renderer:

```
Text prompt → Streaming ODE integration → Frame-by-frame decoding → Real-time HTML update
```

Generation runs on a background thread so the UI stays responsive (the
"Stop" button works mid-generation). The final smoothing pass still
applies once the full trajectory is computed, so the eventual file
matches what the non-streaming path would have produced.

### Running

```bash
# Direct
python gradio_app_streaming.py

# Custom port
python gradio_app_streaming.py --port 7862

# Public sharing
python gradio_app_streaming.py --share

# Via Makefile (see DEPLOYMENT.md for the full Makefile reference)
make run-gradio-streaming
```

The interface is at `http://localhost:7860` by default; if 7860 is
taken, gradio falls back to 7861, 7862, etc.

### Interface controls

- **Input text** — motion description
- **Action duration** — clip length (0.5 – 12 s)
- **Random seeds** — comma-separated seeds for reproducibility
- **CFG scale** — prompt adherence (1.0 – 10.0)
- **Generate Motion (Streaming)** — start frame-by-frame generation
- **Stop Generation** — halt mid-generation; partial result is preserved
- **🔄 Motion Continuation** accordion — see below

### Streaming vs. standard gradio

| | Streaming | Standard |
|---|---|---|
| Total generation time | same | same |
| Visual feedback | progressive frames | spinner until done |
| Stop mid-generation | ✓ | ✗ |
| Memory usage | slightly higher (intermediate buffers) | baseline |
| Final output quality | identical | identical |

### Limitations

- **No FBX streaming** — FBX export only happens at completion
- **Smoothing applies only to the final result** — intermediate streamed
  frames may look very slightly rougher than the final clip
- More moving parts than the standard path; if you hit problems, try the
  non-streaming gradio first to isolate

---

## Motion continuation

Chain multiple short clips into a longer sequence by feeding the JSON
output of one generation back in as the seed for the next. Useful for:

- Extending an existing animation
- Building multi-stage motions ("walk → run → jump") from focused prompts
- Iterative refinement (regenerate just the tail of a sequence)

### Workflow

1. **Generate an initial motion.** Type a prompt, generate, and download
   the resulting JSON file.
2. **Continue.** Expand the **🔄 Motion Continuation** accordion. Upload
   the JSON. Tick **Enable Continuation Mode**. Type a new prompt that
   describes what comes *next*. Generate.

Result: the new clip is appended to the original. Frame counts, root
translation, and pose data all extend cleanly along the time axis.

### Example: walking → running

```
Step 1:  prompt = "A person walking forward at a moderate pace"
         duration = 5 s
         → walking.json   (150 frames)

Step 2:  upload walking.json, enable continuation
         prompt = "then starts running faster"
         duration = 3 s
         → 8-second clip (5 s walking + 3 s running)
```

### JSON format

The continuation feature reads the standard HY-Motion JSON shape:

```json
{
  "frameCount": 150,
  "poses": [[...156 floats per frame...]],
  "trans": [[x, y, z], ...],
  "Rh":    [[rx, ry, rz], ...],
  "text":  "original prompt",
  "timestamp": "...",
  "batchIndex": 0
}
```

Internally `poses` (156-dim per-frame) is converted to `rot6d`
(22 joints × 6 = 132 floats per frame) for concatenation with the new
generation, then converted back when serialised.

### Caveats

- **No conditioning on the last frame.** The continuation generation is
  a fresh run with a new prompt; it doesn't actually consume the prior
  motion's last frame as a starting state. Visual continuity depends on
  the prompts being compatible.
- **No transition smoothing.** The seam between clips can be visible if
  the two motions diverge sharply. Mitigations: shorter durations, more
  similar prompts, or chain via several smaller continuation steps.
- **`keypoints3d` is not regenerated** for the combined output. The
  visualizer recomputes it on the fly; if you consume the JSON
  programmatically and need keypoints, recompute them yourself from
  `poses` + `trans` via the SMPL-H body model.

### Programmatic continuation

```python
import json
from gradio_app_streaming import streaming_generate_motion

for html, files in streaming_generate_motion(
    text="then starts running",
    seeds_csv="42",
    motion_duration=3.0,
    cfg_scale=7.0,
    output_format="json",
    continue_from_json="walking.json",
    continue_mode=True,
):
    # Each yield is an intermediate frame update.
    pass

# The final entry of `files` is the combined motion JSON.
```

---

## REST API server

`streaming_server.py` exposes the same generation engine over HTTP for
embedding into other applications. No gradio UI, no static assets — just
a JSON endpoint.

### Running

```bash
python streaming_server.py --port 8000
```

### Endpoints

The server exposes a `POST /generate` endpoint accepting the same prompt
parameters as the gradio app and returning a JSON body with the SMPL-H
motion data. Refer to `streaming_server.py` for the canonical request
schema (it's small, ~200 lines).

### Use case

This is the entry point Voxta talks to (indirectly, via a child-process
IPC wrapper rather than HTTP). For embedding into a different
application, calling `streaming_server.py` directly avoids spinning up
gradio just to issue a few generations.

---

## Troubleshooting

The most common streaming failures and their fixes:

### "🕒 Generating motion frames..." spinner stuck

Three families of causes, in rough order of likelihood:

**1. Runtime didn't initialize.** Checkpoint files weren't found at
startup. Look for `>>> Loading model from ...` in the console. If
absent, the runtime is `None` and generation will hang.

```bash
# Verify checkpoints exist and config.yml + latest.ckpt are siblings
ls ckpts/tencent/HY-Motion-1.0/
```

If you need a non-default location, point `T2MRuntime`'s `config_path`
to your config and let the bare `ckpt_name="latest.ckpt"` resolve
against its directory automatically (this fork's path-resolution fix —
no symlink workarounds required).

**2. Frame generation thread crashed silently.** Look for tracebacks in
the console output above the spinner. Most often: CUDA OOM under
quantization-disabled paths, or a missing dependency surfacing only when
the streaming variant exercises a code path the standard gradio doesn't.

**3. HTML rendering failed.** The streaming UI re-renders the partial
motion on every frame; a render failure leaves the spinner alive. Watch
for warnings about template / asset paths.

### Frames appear choppy or out of order

Browser-side or system-load issue, not generation. Try:
- A simpler prompt / shorter duration
- A less heavily loaded browser tab
- Disabling other browser extensions that throttle requestAnimationFrame

### Final result differs from the streaming preview

Expected. The final pass applies smoothing that intermediate streamed
frames don't carry. The streamed view shows raw integration output;
saved files are post-smoothing.

### Continuation: "Error loading continuation JSON"

The uploaded file isn't in the expected shape. Verify:
- `frameCount` is present and matches the array lengths
- `poses` is a 2-D array with shape `(frameCount, 156)`
- `trans` is `(frameCount, 3)`
- `Rh` is optional but if present should be `(frameCount, 3)`

### Continuation: motion looks discontinuous at the join

Expected limitation — see "Caveats" above. Use shorter steps and
similar prompts; the more the new prompt deviates from the original
motion's end pose, the more visible the seam.

---

## Performance notes

- **Memory.** Streaming buffers an extra few MB per inflight generation
  (the partial decoded frames). Generally negligible vs. the model itself.
- **CPU.** Threading + queue management adds a small constant overhead
  vs. the non-streaming path; not a bottleneck.
- **GPU.** Identical to the standard path — same forward passes, same
  amount of compute.
- **Network.** Local; no extra network overhead.

---

## Related docs

- [DEPLOYMENT.md](DEPLOYMENT.md) — Docker, Makefile shortcuts, quantization,
  GGUF setup
- [README.md](README.md) — install, library quick-start, model weights,
  fork notice
