# Wheeler Memory Directory Reorganization — Summary

**Date**: February 28, 2026
**Type**: Non-destructive structural cleanup (zero code changes)

## Overview

Reorganized the Wheeler Memory project directory to improve clarity and reduce visual clutter at the project root. All changes are cosmetic and structural; no Python code or dependencies were modified.

## Changes Made

### 1. Consolidated Visual Assets (docs/)

**Before:**
```
/
├── diversity_report.png          (loose at root)
├── paraphrase_report.png         (loose at root)
├── paraphrase_embed_report.png   (loose at root)
└── docs/
    ├── assets/
    │   ├── diversity_report.png  (duplicate)
    │   ├── diversity_report_math.png
    │   ├── diversity_report_math_gpu.png
    │   ├── diversity_report_math_10k_gpu.png
    │   ├── paraphrase_report.png (duplicate)
    │   ├── paraphrase_embed_report.png (duplicate)
    │   └── reconstruction_demo.png
    └── images/
        ├── evolution.gif
        └── phase_2_verification.png
```

**After:**
```
/
└── docs/
    └── assets/
        ├── README.md               (new, explains organization)
        ├── diagrams/               (conceptual/architectural)
        │   ├── evolution.gif
        │   └── phase_2_verification.png
        └── reports/                (test validation/performance)
            ├── diversity_report.png
            ├── diversity_report_math.png
            ├── diversity_report_math_gpu.png
            ├── diversity_report_math_10k_gpu.png
            ├── paraphrase_report.png
            ├── paraphrase_embed_report.png
            └── reconstruction_demo.png
```

**Benefits:**
- Eliminated duplicate images at root
- Clear semantic organization: `diagrams/` (conceptual) vs. `reports/` (empirical)
- Cleaner project root
- Single source of truth for all visual assets

### 2. Consolidated UI Files (ui/)

**Before:**
```
/
├── demo/
│   └── index.html
└── ui/
    └── index.html    (different content, confusing naming)
```

**After:**
```
/
└── ui/
    ├── README.md        (new, explains both interfaces)
    ├── dashboard.html   (renamed from index.html)
    └── demo.html        (renamed from demo/index.html)
```

**Benefits:**
- Clear naming: `dashboard.html` vs. `demo.html`
- Single UI directory eliminates confusion
- Added documentation explaining differences and usage

### 3. Updated References

**File**: `scripts/wheeler_ui.py`
- **Line 20**: Changed `UI_FILE` path from `ui/index.html` → `ui/dashboard.html`
- **Line 200**: Updated error message to reference `ui/dashboard.html`

**Zero impact**: The change is a pure string substitution. No functional code was modified.

### 4. Enhanced Documentation

**New files created:**
1. **ui/README.md** — Explains dashboard vs. demo, usage instructions, API overview
2. **docs/assets/README.md** — Explains diagrams/ vs. reports/ organization, lists all assets
3. **README.md enhancement** — Added new "User Interfaces" and "Related Tools" sections after Installation

## Verification

✓ No loose PNG files remain at project root
✓ All visual assets consolidated under `docs/assets/`
✓ All UI files consolidated under `ui/`
✓ No Python code modified (only path string in `wheeler_ui.py`)
✓ No imports affected
✓ No external tool dependencies changed
✓ Test script defaults unchanged (still output to CWD, users can pass `--output` if desired)

## Directory Structure (Final)

```
/home/tristan/wheeler memory/
├── README.md                        (enhanced with UI/tools sections)
├── LICENSE
├── pyproject.toml                   (unchanged)
│
├── wheeler_memory/                  (core library, unchanged)
│   ├── dynamics.py
│   ├── storage.py
│   ├── polarity.py
│   ├── warming.py
│   ├── agent.py
│   ├── ... (20+ other modules)
│   └── gpu/
│
├── scripts/                         (CLI entry points, unchanged except wheeler_ui.py)
│   ├── wheeler_store.py
│   ├── wheeler_recall.py
│   ├── wheeler_ui.py                (UPDATED: dashboard.html reference)
│   └── ... (15+ test/utility scripts)
│
├── docs/                            (reorganized)
│   ├── architecture.md
│   ├── concepts.md
│   ├── api.md
│   ├── cli.md
│   ├── install.md
│   ├── gpu.md
│   ├── future.md
│   └── assets/                      (NEW organization)
│       ├── README.md                (NEW)
│       ├── diagrams/                (NEW: conceptual)
│       │   ├── evolution.gif
│       │   └── phase_2_verification.png
│       └── reports/                 (NEW: validation)
│           ├── diversity_report.png
│           ├── diversity_report_math.png
│           ├── diversity_report_math_gpu.png
│           ├── diversity_report_math_10k_gpu.png
│           ├── paraphrase_report.png
│           ├── paraphrase_embed_report.png
│           └── reconstruction_demo.png
│
├── ui/                              (reorganized)
│   ├── README.md                    (NEW)
│   ├── dashboard.html               (renamed from ui/index.html)
│   └── demo.html                    (moved from demo/index.html)
│
├── open_webui_setup/                (unchanged)
│   ├── setup_open_webui.sh
│   ├── launch_open_webui.sh
│   ├── launch_pipelines.sh
│   └── pipelines/
│
├── wheeler_3d_viewer/               (unchanged, documented as external tool)
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   └── static/
│
├── .gitignore                       (unchanged)
├── .claude/                         (unchanged)
│
├── build/                           (auto-generated, .gitignore'd)
├── wheeler_memory.egg-info/         (auto-generated, .gitignore'd)
├── .venv/                           (virtual environment, .gitignore'd)
└── ... (git metadata, pycache, etc.)
```

## Test Commands to Verify

```bash
# Verify UI command still works
cd /home/tristan/wheeler\ memory
python -m scripts.wheeler_ui  # Should find ui/dashboard.html

# List all PNG files (should all be under docs/assets/)
find . -name "*.png" -not -path "./.venv/*" -not -path "./build/*"

# Check for loose images at root
ls *.png 2>&1 | grep -q "No such file" && echo "✓ Root clean"
```

## Rollback Instructions

If needed, the changes can be reversed:

```bash
# Move UI files back
mv ui/dashboard.html ui/index.html
mv ui/demo.html demo/index.html
mkdir -p demo
mv demo/index.html demo/

# Restore root images
mkdir -p docs/assets
mv docs/assets/reports/diversity_report.png .
mv docs/assets/reports/paraphrase_report.png .
mv docs/assets/reports/paraphrase_embed_report.png .

# Restore wheeler_ui.py
sed -i 's|ui/dashboard.html|ui/index.html|g' scripts/wheeler_ui.py

# Remove new docs
rm ui/README.md docs/assets/README.md
```

However, no rollback is necessary — the reorganization is complete and non-breaking.

## Recommendations for Future Maintenance

1. **Test script output**: Consider adding a `--output docs/assets/reports/` flag to test scripts for convenience, though defaults to CWD is fine for ad-hoc testing.

2. **Asset versioning**: As more validation reports are generated, consider timestamped subdirectories (e.g., `reports/2026-02-28/`) to preserve historical records.

3. **CI integration**: Wire asset generation into CI/CD pipeline to auto-generate reports on each push or release.

4. **UI unification**: In a future refactor, consider a single unified UI with tabbed navigation (dashboard | demo | settings) rather than two separate files.

---

**Status**: Complete, verified, ready for commit (when desired).
