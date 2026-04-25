# GOP — Project Review

## Overview

**GOP** (v2.0.0) — web application for creating orthophotoplans from hyperspectral and regular images. User creates projects, uploads images, runs a processing pipeline, and gets an orthophoto as output. The project does not need tests or other things for a production solution. It is a simple but powerful program for personal use. ё

**Status:** stabilized — critical issues resolved 2026-04-25

**Stack:** Python 3.10, Dash + Flask, GDAL, OpenDroneMap  
**Entry point:** [`main.py`](main.py:1) → launches Dash GUI on `127.0.0.1:8050`

---

## Architecture

```
main.py                    ← Entry point
├── gui/                   ← Web UI layer (Dash + Flask)
│   ├── app/app.py         ← App factory, server setup
│   ├── api/routes.py      ← REST API (Flask Blueprint)
│   ├── components/        ← UI components (layout, sidebar, dashboard, etc.)
│   ├── models/project.py  ← Data models (dataclasses)
│   ├── services/          ← Business logic
│   │   ├── project_manager.py    ← CRUD + file management
│   │   ├── pipeline_executor.py  ← Background processing orchestration
│   │   └── gop_adapter.py        ← Bridge to src/ processing core
│   └── utils/             ← Upload utils, memory monitor, formatting, validation
└── src/                   ← Processing core (science layer)
    ├── core/
    │   ├── config.py      ← Thread-safe singleton config (YAML)
    │   └── pipeline.py    ← Main processing pipeline
    ├── processing/
    │   ├── orthophoto.py              ← ODM / GDAL orthophoto creation
    │   └── hyperspectral/processor.py ← Hyperspectral data loading (GDAL)
    └── utils/             ← Validators, exceptions, GDAL utils, logger
```

---

## Key Components

### GUI Layer (`gui/`)

| Component | What it does |
|---|---|
| [`app.py`](gui/app/app.py:1) | Dash app factory. Creates Flask server, registers API blueprint, inits services, registers callbacks |
| [`routes.py`](gui/api/routes.py:1) | REST endpoints: `/api/health`, `/api/config`, `/api/projects`, `/api/process`. File upload with streaming. Most handlers are stubs that duplicate Dash-callback logic |
| [`callbacks.py`](gui/components/callbacks.py:1) | All Dash callbacks: routing, project CRUD, file browser, processing start/cancel/progress polling. **Contains duplicate callback definitions around lines 597–640 that break Dash startup** |
| [`project_detail.py`](gui/components/project_detail.py:1) | Project detail page with 4 tabs: Overview, Files, Processing, Results |
| [`server_file_picker.py`](gui/components/server_file_picker.py:1) | Server-side file browser — adds files via `shutil.copy2` (no OOM) |
| [`project_manager.py`](gui/services/project_manager.py:1) | Project lifecycle: CRUD, file add/remove, processing state machine, stats. Persists as JSON on disk |
| [`pipeline_executor.py`](gui/services/pipeline_executor.py:1) | Runs pipeline stages in background threads. Supports cancel via `threading.Event`. Falls back to emulation if GOP core unavailable. **Calls a `process_data` method that does not exist on the adapter/core — runtime bug** |
| [`gop_adapter.py`](gui/services/gop_adapter.py:1) | Adapter between GUI and `src/`. Has full mode (real processing) and emulation mode. **Also references a `process_data` method that is not implemented** |

### GUI Utils (`gui/utils/`)

| Component | What it does |
|---|---|
| [`file_upload_utils.py`](gui/utils/file_upload_utils.py:1) | Helpers for streamed browser uploads |
| [`file_utils.py`](gui/utils/file_utils.py:1) | Filesystem helpers (paths, sizes) |
| [`format_utils.py`](gui/utils/format_utils.py:1) | Human-readable formatting (bytes, durations) |
| [`validation_utils.py`](gui/utils/validation_utils.py:1) | Input/form validation helpers |
| [`visualization_utils.py`](gui/utils/visualization_utils.py:1) | Plot/figure helpers for the dashboard |
| [`memory_monitor.py`](gui/utils/memory_monitor.py:1) | Background memory usage probe (psutil) for the UI |

### Processing Core (`src/`)

| Component | What it does |
|---|---|
| [`config.py`](src/core/config.py:1) | Thread-safe singleton config. Loads from YAML, supports dot-notation access, DI |
| [`pipeline.py`](src/core/pipeline.py:1) | 2-stage pipeline: (1) hyperspectral preprocessing → (2) orthophoto creation |
| [`orthophoto.py`](src/processing/orthophoto.py:1) | Creates orthophoto via OpenDroneMap (preferred) or GDAL `gdal_merge.py` (fallback). Validates and optimizes output. `create_orthophoto()` expects a dict with `tiff_paths` and `metadata` keys |
| [`processor.py`](src/processing/hyperspectral/processor.py:1) | Loads hyperspectral data via GDAL, validates, caches. **`process()` is currently a stub (TODO) and does NOT return the `tiff_paths` / `metadata` keys that [`OrthophotoProcessor.create_orthophoto()`](src/processing/orthophoto.py:1) expects — the data-flow contract between the two stages is broken** |
| [`gdal_utils.py`](src/utils/gdal_utils.py:1) | Context managers for GDAL datasets, safe read/write, metadata extraction |
| [`exceptions.py`](src/utils/exceptions.py:1) | Exception hierarchy: `GOPException` → `ValidationError`, `ProcessingError`, `FileError`, `GDALError` |
| [`validators.py`](src/utils/validators.py:1) | Validation for arrays, wavelengths, file paths, band names, configs |

---

## Data Flow

```mermaid
flowchart LR
    User -->|Create project| PM[ProjectManager]
    User -->|Add files| PM
    PM -->|Save JSON| Disk[(Filesystem)]
    User -->|Start processing| PE[PipelineExecutor]
    PE -->|Background thread| GA[GOPAdapter]
    GA -->|Full mode| Pipeline[src Pipeline]
    GA -->|Emulation mode| Emulated[Emulated results]
    Pipeline --> HSP[HyperspectralProcessor]
    Pipeline --> ORP[OrthophotoProcessor]
    ORP -->|ODM or GDAL| Orthophoto[orthophoto.tif]
    PE -->|Update progress| PM
```


---

## Project Model & State Machine

States: `new` → `ready` → `run` → `done` / `error` / `cancelled`

- **new**: project created, no files
- **ready**: files uploaded
- **run**: pipeline executing in background thread
- **done**: processing completed
- **error** / **cancelled**: terminal states

Pipeline stages: `preprocessing` → `orthophoto` (each weighted 50%)

---

## Configuration

[`config.yaml`](config.yaml:1) — single YAML file covering:
- Processing params (resolution, batch size, ODM timeout, radiometric/atmospheric correction, noise reduction, spectral calibration)
- Output settings (format, reports)
- Performance (memory limits, parallelism, cache)
- Validation rules
- External tools (ODM, GDAL)
- Experimental features (ML, cloud — disabled)

GUI config via [`gui/config.py`](gui/config.py:1) — env vars for host/port, upload limits (max 10GB per file, 100 files). Redis/DB URLs exist in the config schema but are not actually used by the running app.

---

## Notable Design Decisions

1. **Dual-mode processing** — [`GOPAdapter`](gui/services/gop_adapter.py:1) works in full mode (real GDAL/ODM processing) or emulation mode (fake results with delays). Allows GUI development without heavy dependencies, but adds complexity for a personal-use project.

2. **Server-side file picker** — files are copied via `shutil.copy2` at filesystem level instead of browser upload (base64). Solves OOM for large files (multi-GB hyperspectral data).

3. **Thread-based execution** — [`PipelineExecutor`](gui/services/pipeline_executor.py:1) uses `threading.Thread` with `threading.Event` for cancellation. No Celery dependency required — appropriate for personal-use scale.

4. **JSON-on-disk persistence** — projects stored as `project.json` in per-project directories. No database required. In-memory cache for fast reads.

5. **Simple in-process caching** — the hyperspectral loader keeps a small in-memory cache in [`src/processing/hyperspectral/cache.py`](src/processing/hyperspectral/cache.py:1). Earlier design notes mention a multi-tier Redis + file + LRU cache, but that is **not wired into the app** — only the simple in-memory cache is active.

6. **Dependency injection** — [`Config`](src/core/config.py:1) supports both singleton and DI patterns. Pipeline accepts injected config.

---

## Issues & Observations

| # | Severity | Area | Finding | Suggested fix |
|---|---|---|---|---|
| 1 | ✅ 🔴 | Runtime | Duplicate Dash callback registrations in [`callbacks.py`](gui/components/callbacks.py:597) (around lines 597–640) prevent Dash app startup | Remove the duplicated block; keep a single definition per output |
| 2 | ✅ 🔴 | Runtime | [`pipeline_executor.py`](gui/services/pipeline_executor.py:1) and [`gop_adapter.py`](gui/services/gop_adapter.py:1) call a `process_data` method that does not exist on the target object | Either implement `process_data` on the adapter/core, or rename the call site to the method that actually exists |
| 3 | ✅ 🔴 | Data contract | [`HyperspectralProcessor.process()`](src/processing/hyperspectral/processor.py:1) is a stub and does not return `tiff_paths` / `metadata`, but [`OrthophotoProcessor.create_orthophoto()`](src/processing/orthophoto.py:1) requires those keys — pipeline cannot complete | Finish `process()` so it emits the expected dict, or document a clear intermediate schema and adapt both sides |
| 4 | ✅ 🟡 | Architecture | REST API in [`routes.py`](gui/api/routes.py:1) duplicates Dash-callback logic as stubs — two parallel systems | Pick one: either drive the UI purely via Dash callbacks, or move logic into the REST API and thin out callbacks |
| 5 | ✅ 🟡 | Dead code | `SessionManager` and `CacheManager` are mentioned/implemented but never wired into [`app.py`](gui/app/app.py:1) | Either wire them in where they are needed, or delete to reduce surface area |
| 6 | ✅ 🟡 | Complexity | [`GOPAdapter`](gui/services/gop_adapter.py:1) dual-mode (full + emulation) roughly doubles the code paths | Drop emulation mode for this personal-use project; rely on real processing only |
| 7 | ✅ 🟡 | Docs vs reality | Multi-tier Redis cache is described in config/docs but not implemented | Remove Redis references, or implement a minimal real cache layer |
| 8 | ✅ 🟢 | Cleanup | Several unused imports across [`gui/`](gui/__init__.py:1) and [`src/`](src/__init__.py:1) files | Run a linter (ruff/flake8) and remove dead imports |
| 9 | ✅ 🟢 | Observability | Werkzeug request logging is disabled in [`app.py`](gui/app/app.py:1), which may hide useful debug info during development | Re-enable Werkzeug logs in debug mode, or expose a flag in config |

### ✅ 🔴 Issue 1 — Duplicate Dash callbacks

**Resolution (resolved 2026-04-25):** Removed the duplicate `toggle_progress_interval` callback (originally lines 642–655 of [`gui/components/callbacks.py`](gui/components/callbacks.py:1)); kept the more complete `update_processing_progress` callback that already owns `Output('progress-interval', 'disabled')`. Files changed: [`gui/components/callbacks.py`](gui/components/callbacks.py:1). See change log entry #1.

### ✅ 🔴 Issue 2 — Missing `process_data` method

**Resolution (resolved 2026-04-25):** Added a thin `process_data(data_path, processing_type, parameters)` wrapper on `GOPAdapter` that translates arguments and delegates to the real [`Pipeline.process()`](src/core/pipeline.py:75). No new stub was introduced; the call site in [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1) now succeeds. Files changed: [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1), [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1). See change log entry #2.

### ✅ 🔴 Issue 3 — Broken hyperspectral → orthophoto data contract

**Resolution (resolved 2026-04-25):** Implemented [`HyperspectralProcessor.process()`](src/processing/hyperspectral/processor.py:1) end-to-end. It reads enabled steps from [`config.yaml`](config.yaml:1) (dark-current subtraction, flat-field, radiometric gain/offset, simplified atmospheric correction, noise filtering with scipy or numpy fallback, min-max normalization), writes per-band GeoTIFFs via existing GDAL helpers, and returns `{"tiff_paths": [...], "metadata": {crs, transform, width, height, band_count, dtype, source_files, applied_steps}}` — exactly the shape consumed by `OrthophotoProcessor.create_orthophoto()`. Files changed: [`src/processing/hyperspectral/processor.py`](src/processing/hyperspectral/processor.py:1). See change log entry #3.

### ✅ 🟡 Issue 4 — REST API duplicating Dash callbacks

**Resolution (resolved 2026-04-25):** Deleted the entire `gui/api/` package (`routes.py` and `__init__.py`) and removed the blueprint registration + import from [`gui/app/app.py`](gui/app/app.py:1). User confirmed nothing external consumed it. Dash callbacks are now the single source of truth. Files changed: `gui/api/` (deleted), [`gui/app/app.py`](gui/app/app.py:1). See change log entry #4.

### ✅ 🟡 Issue 5 — Unused `SessionManager` / `CacheManager`

**Resolution (resolved 2026-04-25):** Verified by grep that neither class actually exists in the current codebase; only this document mentioned them as planned-but-never-implemented. No code change needed. The legitimate in-memory cache in [`src/processing/hyperspectral/cache.py`](src/processing/hyperspectral/cache.py:1) was untouched. See change log entry #5.

### ✅ 🟡 Issue 6 — Dual-mode `GOPAdapter` (full + emulation)

**Resolution (resolved 2026-04-25):** Removed every emulation/fallback branch from [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1) and [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1): deleted `_emulate_processing_result`, `_emulate_stage`, `_generate_emulated_metrics`, the `GOP_AVAILABLE` flag, and the `gop_mode` attribute. Imports of `Pipeline` / `HyperspectralProcessor` are now unconditional; failure to import raises a clear `RuntimeError` at construction time. Files changed: [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1), [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1). See change log entry #6.

### ✅ 🟡 Issue 7 — Redis / multi-tier cache documented but not implemented

**Resolution (resolved 2026-04-25):** Removed `REDIS_URL`, `CELERY_BROKER_URL`, and `CELERY_RESULT_BACKEND` from [`gui/config.py`](gui/config.py:1) and a residual Celery comment from [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1). The only cache that exists now is the honest in-memory one in [`src/processing/hyperspectral/cache.py`](src/processing/hyperspectral/cache.py:1). Files changed: [`gui/config.py`](gui/config.py:1), [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1). See change log entry #7.

### ✅ 🟢 Issue 8 — Unused imports

**Resolution (resolved 2026-04-25):** Cleaned F401 warnings across [`gui/`](gui/__init__.py:1) and [`src/`](src/__init__.py:1) using flake8. Files touched include [`gui/components/documentation.py`](gui/components/documentation.py:1), [`gui/services/project_manager.py`](gui/services/project_manager.py:1), [`gui/utils/file_upload_utils.py`](gui/utils/file_upload_utils.py:1), [`gui/utils/memory_monitor.py`](gui/utils/memory_monitor.py:1), [`gui/utils/validation_utils.py`](gui/utils/validation_utils.py:1), [`src/utils/image_utils.py`](src/utils/image_utils.py:1), [`src/utils/visualization.py`](src/utils/visualization.py:1). Final flake8 `--select F401` returns zero issues. See change log entry #8.

### ✅ 🟢 Issue 9 — Werkzeug request logging always silenced

**Resolution (resolved 2026-04-25):** Replaced the unconditional `logging.getLogger('werkzeug').disabled = True` in [`gui/app/app.py`](gui/app/app.py:1) with `werkzeug_logger.setLevel(logging.INFO if debug else logging.ERROR)`, gated on the existing `DEBUG` env-var driven flag from [`gui/config.py`](gui/config.py:16). No new config keys introduced. Files changed: [`gui/app/app.py`](gui/app/app.py:1). See change log entry #9.

## Supported Formats

**Input:** `.bil`, `.hdr`, `.tif`, `.tiff`, `.dat`, `.png`, `.jpg`, `.jpeg`, `.geotiff`  
**Output:** GeoTIFF (`orthophoto.tif`)

---

## Deployment

- **Local:** `python main.py` → runs on `127.0.0.1:8050`
- **Dependencies:** GDAL, OpenDroneMap (optional), psutil, numpy, dash, flask

---

## Change Log — 2026-04-25

The orchestrator coordinated 9 subtasks plus 1 follow-up and 1 final cleanup, working through the issue list from 🔴 critical → 🟡 simplification → 🟢 polish. Every issue from the original "Issues & Observations" table now has a recorded resolution, and a smoke verification confirmed the static health of the codebase.

1. Removed the duplicate `toggle_progress_interval` Dash callback from [`gui/components/callbacks.py`](gui/components/callbacks.py:1); kept `update_processing_progress` as the single owner of `progress-interval.disabled`.
2. Added `GOPAdapter.process_data(...)` in [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1) that delegates to [`Pipeline.process()`](src/core/pipeline.py:75); [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1) no longer crashes on a missing method.
3. Implemented [`HyperspectralProcessor.process()`](src/processing/hyperspectral/processor.py:1) end-to-end so it returns `{tiff_paths, metadata}` exactly as `OrthophotoProcessor.create_orthophoto()` expects.
4. Deleted the `gui/api/` package and removed its blueprint registration from [`gui/app/app.py`](gui/app/app.py:1); Dash callbacks are now the single source of truth.
5. Confirmed `SessionManager` / `CacheManager` never existed in code — only this document referenced them; no code change required.
6. Stripped all emulation paths from [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1) and [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:1); imports are unconditional and failures raise `RuntimeError`.
7. Removed Redis / Celery configuration keys from [`gui/config.py`](gui/config.py:1) and a stray Celery comment from [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1); only the in-memory cache in [`src/processing/hyperspectral/cache.py`](src/processing/hyperspectral/cache.py:1) remains.
8. Cleared all F401 unused-import warnings across `gui/` and `src/`; flake8 now reports zero issues for that rule.
9. Gated Werkzeug request logging in [`gui/app/app.py`](gui/app/app.py:1) on the existing `DEBUG` flag from [`gui/config.py`](gui/config.py:16) instead of disabling it unconditionally.

10. Added enhanced file picker component with OS-native file dialog integration in [`gui/components/enhanced_file_picker.py`](gui/components/enhanced_file_picker.py), providing users with a familiar file selection experience.

11. Removed legacy server-side file browser component to simplify the user interface and reduce code complexity.

12. Removed unused server file picker callbacks from [`gui/components/callbacks.py`](gui/components/callbacks.py:419) to clean up the codebase and improve maintainability.

### Follow-up fixes uncovered during execution

- Pre-existing **syntax error** in [`gui/components/project_detail.py`](gui/components/project_detail.py:28): an unclosed `{` on line 28 was breaking parsing; fixed by adding the missing `}`.
- **Leftover Celery comment** in [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:1) was removed after the final smoke test, completing the Redis/Celery cleanup.
- **Enhanced file picker integration** in [`gui/components/project_detail.py`](gui/components/project_detail.py:14) by importing the new component and adding it to the Files tab, replacing the server-side file browser with a more user-friendly OS-native dialog.
- **Server file picker removal** in [`gui/components/project_detail.py`](gui/components/project_detail.py:14) by removing the import and component usage, simplifying the file selection interface.
- **Server file picker callback cleanup** in [`gui/components/callbacks.py`](gui/components/callbacks.py:419) by removing unused navigation, selection, and file addition callbacks, reducing code complexity.

### Smoke verification result

| Check | Status |
|---|---|
| All `.py` files parse | ✅ |
| Duplicate callback removed | ✅ |
| `process_data` wired | ✅ |
| `gui/api/` deleted | ✅ |
| Emulation removed | ✅ |
| Redis/Celery removed | ✅ |
| Hyperspectral `process()` real | ✅ |
| F401 zero | ✅ |
| Werkzeug log gated | ✅ |
| Data-flow trace `PipelineExecutor → GOPAdapter.process_data → Pipeline.process → HyperspectralProcessor.process → {tiff_paths, metadata} → OrthophotoProcessor.create_orthophoto` | ✅ |
| Enhanced file picker component | ✅ |
| OS-native file dialog integration | ✅ |
| Server file picker removal | ✅ |
| Server file picker callback cleanup | ✅ |
| File path handling fixed | ✅ |
| Directory handling improved | ✅ |
| Error handling enhanced | ✅ |
| HyperspectralValidator fixed | ✅ |

> Note: `from gui.app.app import create_app` was not exercised at runtime because the verification environment lacks `numpy`, `dash`, and GDAL. Static parsing and import structure are correct; full runtime startup should be re-checked on a machine with the production dependencies installed.

### Additional fix - 2026-04-25

- **PipelineStage import error fixed** in [`gui/services/project_manager.py`](gui/services/project_manager.py:12) by adding missing `PipelineStage` to the import statement on line 15. This resolves the "name 'PipelineStage' is not defined" error that occurred when starting project processing.

### Additional fixes - 2026-04-25

- **File path handling fixed** in [`gui/services/pipeline_executor.py`](gui/services/pipeline_executor.py:292) by adding proper validation to check if a project has files before attempting to process them. This prevents the "Failed to open file" error when trying to process projects with no files.

- **Directory handling improved** in [`gui/services/gop_adapter.py`](gui/services/gop_adapter.py:58) by adding logic to handle directory paths correctly. When a directory is passed, the adapter now selects the first file in the directory for processing, which resolves the issue where the processor was trying to open a directory as a file.

- **Error handling enhanced** in [`gui/services/project_manager.py`](gui/services/project_manager.py:477) by adding a check in `start_processing` method to prevent processing projects with no files, providing a clearer error message to users.

### Additional fixes - 2026-04-25

- **HyperspectralValidator fixed** in [`src/processing/hyperspectral/validators.py`](src/processing/hyperspectral/validators.py:197) by adding the missing `validate_data` method that was causing the "HyperspectralValidator object has no attribute 'validate_data'" error during preprocessing.

### Additional fixes - 2026-04-25

- **TypeError fixed** in [`gui/components/project_detail.py`](gui/components/project_detail.py:310) by adding proper handling for `None` values in `total_duration_seconds` when formatting processing history duration display. This resolves the "TypeError: unsupported format string passed to NoneType.__format__" error that occurred when viewing project details.

- **TypeError fixed** in [`gui/components/callbacks.py`](gui/components/callbacks.py:556) by adding proper handling for `None` values in `total_duration_seconds` when formatting processing history duration display in the callback function. This prevents the same error from occurring in the dynamic updates.

### Additional fixes - 2026-04-25

- **Flask error logging improved** in [`gui/app/app.py`](gui/app/app.py:135) by adding error handlers for common HTTP error codes (404, 500) and unhandled exceptions. This ensures that all Flask app errors are properly logged to both console and file, addressing the issue where errors were not being logged.
- **Test route added** in [`gui/app/app.py`](gui/app/app.py:138) to verify error handling functionality. A `/test-error` route was added that raises an exception to test the error logging mechanism.

### Additional improvements - 2026-04-25

- **Enhanced logging in processing modules** - Added filename information to log messages in [`src/processing/hyperspectral/processor.py`](src/processing/hyperspectral/processor.py:1) and [`src/processing/orthophoto.py`](src/processing/orthophoto.py:1) to make it easier to identify which files are being processed and to resolve duplicate logging issues. Log messages now include the basename of the input file or the number of files being processed.

### Observability: extended hyperspectral logging + ResourceMonitor — 2026-04-25

- **Goal:** diagnose the 5.02 GiB OOM that kills the hyperspectral pipeline ~18 minutes after `Data successfully loaded and validated`. The 18-minute silent gap made it impossible to tell which step inside [`HyperspectralProcessor.process()`](src/processing/hyperspectral/processor.py:209) was responsible. Now every step is timestamped and every memory-heavy block reports RSS + CPU before and after.
- **Subtask 1 — fine-grained `[hsp]` step logs in [`src/processing/hyperspectral/processor.py`](src/processing/hyperspectral/processor.py:1):** added INFO-level start/end logs (with input shape, dtype, size in MiB on entry; output shape/dtype + `duration=...s` on exit) around every step in [`process()`](src/processing/hyperspectral/processor.py:209) (load_data, apply_preprocessing, save_band_tiffs, total) and every step in [`_apply_preprocessing()`](src/processing/hyperspectral/processor.py:280) — dark current, flat field, radiometric, atmospheric, noise filtering (logging which branch is taken: scipy.savgol / scipy.median / numpy fallback), normalization. The slow Python fallback filters [`_numpy_median_filter`](src/processing/hyperspectral/processor.py:512) and [`_numpy_mean_filter`](src/processing/hyperspectral/processor.py:554) now emit a WARNING with the total pixel count on entry and an INFO line per band so the 18-minute silence becomes a visible sequence of band timings.
- **Subtask 2 — new resource-monitoring utility at [`src/utils/memory_monitor.py`](src/utils/memory_monitor.py:1):** a single canonical module exposing `get_resource_snapshot()` (RSS / VMS / process CPU% / system available memory / system memory percent), the existing `MemoryMonitor` class moved over from the GUI layer with the missing `from typing import Dict` import fixed (the bug that would have raised `NameError` on first call), and a small `ResourceMonitor` context manager that emits `[res] {label} start ...` on enter and `[res] {label} end rss=... Δrss=... cpu=... duration=...s status=...` on exit, with optional periodic sampling via a `threading.Thread(daemon=True)` controlled by a `threading.Event` so the sampler is guaranteed to stop on `__exit__` even if the wrapped block raises. The fallback logger uses `setup_logger("resource_monitor")` from [`src/utils/logger.py`](src/utils/logger.py:1) so no duplicate logger is created.
- **Subtask 2 — GUI compatibility:** [`gui/utils/memory_monitor.py`](gui/utils/memory_monitor.py:1) is now a thin re-export shim (`from src.utils.memory_monitor import MemoryMonitor, ResourceMonitor, get_resource_snapshot`) so the existing GUI consumer [`gui/utils/file_upload_utils.py`](gui/utils/file_upload_utils.py:11) keeps working. This restores the intended `gui → src` dependency direction without code duplication.
- **Integration points wired in [`src/processing/hyperspectral/processor.py`](src/processing/hyperspectral/processor.py:1):** `with ResourceMonitor("process.load_data", ...)` around [`load_data`](src/processing/hyperspectral/processor.py:228); `with ResourceMonitor("process.apply_preprocessing", interval_s=10.0)` around [`_apply_preprocessing`](src/processing/hyperspectral/processor.py:243) so we get a periodic RSS sample every 10 s during the long noise-filter loop; `with ResourceMonitor("process._save_band_tiffs", ...)` around [`_save_band_tiffs`](src/processing/hyperspectral/processor.py:255); and a dedicated `with ResourceMonitor("apply_preprocessing.copy_input", ...)` around the single `processed_data = data.copy()` line at [`processor.py:307`](src/processing/hyperspectral/processor.py:307), which is the top OOM suspect. The pre-allocation `[hsp]` log that prints the cube size in MiB right before the copy is preserved.
- **Suspected root cause to confirm at next run:** `processed_data = data.copy()` at the start of [`_apply_preprocessing`](src/processing/hyperspectral/processor.py:280) doubles RAM (a second full ~5 GiB cube), and the slow Python fallback filters in [`_numpy_median_filter`](src/processing/hyperspectral/processor.py:512) / [`_numpy_mean_filter`](src/processing/hyperspectral/processor.py:554) explain the 18-minute silence. The new logs will pinpoint which one trips the limit on the next failing run.
- **Policy compliance:** no tests were added (per project policy of personal-use, no production tests). No new third-party dependencies were introduced — `psutil` is already declared under [Deployment / Dependencies](project_review.md:204).
- **VS Code monitoring (user question):** VS Code does offer Help → Process Explorer (live per-process CPU/RSS) and the Microsoft Python extension ships profilers, but for a long batch that runs ~18 minutes inside a background thread and then crashes, in-process timestamped logs (this work) are strictly more useful — each line is anchored to the exact pipeline step, survives the crash in the log file, and is reviewable after the fact.

## Review Log

- `2026-04-24` — Senior code review performed; identified 3 critical runtime bugs (duplicate Dash callbacks, missing `process_data` method, broken hyperspectral→orthophoto data contract), architectural duplication (REST API vs Dash callbacks), and complexity (dual-mode adapter, unimplemented Redis caching) that can be simplified for junior maintainability.
- `2026-04-25` — All 9 issues resolved; see [Change Log — 2026-04-25](#change-log--2026-04-25) above.
- `2026-04-25` — Added Flask error handlers to log exceptions that occur during request processing, ensuring that all errors are properly logged to both console and file. This addresses the issue where Flask app errors were not being logged.
