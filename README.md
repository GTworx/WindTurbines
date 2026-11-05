# BladeRunner — Wind Turbine Blade Defect Inspector Agent

An event-driven, multi-agent computer-vision framework for wind turbine blade
defect inspection, implemented from the specification in
[`Wind Turbine Blade Defect Inspector Agent.md`](./Wind%20Turbine%20Blade%20Defect%20Inspector%20Agent.md).

It automates **defect-taxonomy learning**, **blade surface inspection**,
**incremental re-examination**, **stateful checkpointing**, and **reporting**
(Markdown wiki + static HTML dashboard).

```
                     ┌────────────────────────────┐
                     │     OrchestratorAgent      │
                     │ (queue · state · resume)   │
                     └──┬───────┬───────────┬─────┘
          ┌─────────────┤       │           ├────────────────┐
          ▼             ▼       ▼           ▼                ▼
   DefectTrainingAgent  BladeInspectorAgent  ReexamineAgent  ReportingAgent
   (taxonomy)           (vision inspection)  (delta re-run)  (wiki+HTML+overlays)
```

## Architecture (from the spec)

| Agent | Spec § | Responsibility |
| --- | --- | --- |
| `OrchestratorAgent` | 2.1 | Coordinate, schedule, manage state / resume, enforce chunking |
| `DefectTrainingAgent` | 2.2 | Learn defect taxonomy from `Defects/`, bump taxonomy version |
| `BladeInspectorAgent` | 2.3 | Inspect blades: metadata + chunked vision inference |
| `ReexamineAgent` | 2.4 | Re-run inference for new/delta defect classes on cached images |
| `ReportingAgent` | 2.5 | Wiki pages, annotated overlays, static HTML dashboard |

Also implemented: token-bounded image chunking (§2.1/2.3), structured JSONL
logging (§3.1), checkpoint state + image-hash idempotency (§3.2), exponential
backoff + circuit breaker (§4.4), severity scale 1–5 (§4.1), and a strict
`metadata.csv` schema contract with header aliases (§4.2).

## Install

Requires Python ≥ 3.9.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e .            # core deps: pillow, numpy, pandas + CLI entry point
pip install -e ".[dev]"     # + pytest
pip install -e ".[local]"   # + torch/transformers/ultralytics for the local model backend
```

`pip install -e .` registers the `bladerunner` command.

> The **mock** backend (default) runs the entire pipeline with no GPU and no
> model download — perfect for development and testing. The **local** backend
> (`.[local]`) uses a local Qwen-VL model + optional YOLO detector.

## Usage

### 1. Build / refresh the defect taxonomy
Scans `Defects/` (subfolders by class, or flat files named by defect) and
updates `defect_taxonomy.json`, bumping the version.

```bash
bladerunner train
```

### 2. Inspect a turbine
Turbine layout: `Turbine_X/Blade_A/{metadata.csv, images/*.jpg}`.

```bash
bladerunner inspect --turbine-dir G:/MyDrive/WindTurbineImages/Turbine_T104 \
    --confidence-threshold 0.75 --backend mock --run-id T104
```

### 3. Run the full pipeline (train → inspect → report)
```bash
bladerunner run --turbine-dir Turbine_Demo --backend mock --run-id demo
```
Outputs:
- `Runs/<run_id>/results.json`
- `Runs/<run_id>/annotated/<blade>/<image>.annotated.<ext>`
- `Runs/<run_id>/dashboard/index.html`  ← **the dashboard**
- `Wiki/Turbines/<Turbine_ID>/index.md` + per-blade pages

### 4. Re-examine for a new defect class
```bash
bladerunner reexamine --turbine-dir Turbine_Demo --backend mock \
    --delta-classes micro_pitting --prior-results Runs/demo/results.json
```

### 5. Other
```bash
bladerunner status                          # show checkpoint state
bladerunner report --results Runs/demo/results.json   # regenerate wiki + dashboard
bladerunner init-config --output bladerunner.json     # write example config
```

Pass `--config bladerunner.json --project-root .` before the subcommand to use
a custom config and project root.

## Demo (no turbine data required)

```bash
.venv\Scripts\python tools\make_demo_turbine.py
.venv\Scripts\bladerunner --project-root . run --turbine-dir Turbine_Demo --backend mock --run-id demo
```
Then open `Runs/demo/dashboard/index.html` in a browser.

> ⚠️ The **mock** backend produces *synthetic, deterministic* detections for
> exercising the pipeline — it is **not** a real defect detector. Use
> `--backend local` with `.[local]` installed for genuine inspection.

## Configuration

Defaults live in `bladerunner/config.py`. Override via a JSON config file:

```json
{
  "backend": "local",
  "local": { "vlm_model": "Qwen/Qwen2-VL-7B-Instruct", "detector": "yolo", "device": "auto" },
  "chunking": { "images_per_chunk": 6, "max_chunk_tokens": 8000 },
  "inspection": { "confidence_threshold": 0.75 },
  "paths": { "defects_dir": "Defects", "logs_dir": "Logs", "wiki_dir": "Wiki", "runs_dir": "Runs" }
}
```

## Local vision backend

Set `"backend": "local"` (or `--backend local`). `LocalVisionModel`:
1. Runs an optional **YOLO** detector (ultralytics) for bounding boxes.
2. For each box (or the full image if no detector), prompts a **Qwen-VL**
   multimodal LLM to classify the defect, grade severity 1–5, and describe it.
3. Falls back gracefully (full-image classification) if a detector is missing.

## Project layout

```
bladerunner/
├── cli.py            # `bladerunner` command
├── orchestrator.py   # OrchestratorAgent (train → inspect → report)
├── agents/           # training, inspector, reexamine, reporting, base
├── vision/           # base (ABC), mock, local backends
├── config.py         # Config + defaults
├── models.py         # Detection, BladeReport, TurbineReport, ...
├── taxonomy.py       # defect_taxonomy.json management + severity scale
├── metadata.py       # metadata.csv schema contract + alias mapping
├── chunking.py       # token-bounded image chunking
├── state.py          # checkpoint + image-hash idempotency
├── rate_limit.py     # exponential backoff + circuit breaker
├── overlay.py        # annotated image generation (PIL)
├── events.py         # in-process event bus
└── logging_setup.py  # structured JSONL logging
```

## Testing

```bash
.venv\Scripts\python -m pytest
```

## Spec items intentionally left as stubs / TODO
- **Human-in-the-loop** feedback (spec §4.5): the dashboard renders corrections
  data but the write-back loop to the Training Agent is not yet wired.
- **Defect-model baseline** (spec §5.1): defaults to VLM + optional YOLO; a
  dedicated fine-tuned detector (YOLOv11/RT-DETR) is not included.
- **Filesystem watcher** (spec §2.2 trigger): taxonomy build is on-demand via
  `bladerunner train` rather than a persistent `watchdog` process.
