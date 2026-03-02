# Wheeler Memory User Interfaces

This directory contains interactive web-based interfaces for Wheeler Memory.

## Dashboard (`dashboard.html`)

The main Wheeler Memory control dashboard. This is the primary UI for:
- Storing new memories
- Recalling memories by similarity
- Viewing active memory state and temperatures
- Consolidating memories (sleep)
- Forgetting memories

**Launch**: Use the `wheeler-ui` command (entry point in `pyproject.toml`):
```bash
wheeler-ui
```

This starts a local HTTP server on `http://localhost:7437` and opens the dashboard automatically.

## Demo (`demo.html`)

An interactive demonstration and educational walkthrough of Wheeler Memory. This page showcases:
- Core concepts (CA dynamics, attractors, temperature)
- Visual evolution animations
- Example memory formation and recall
- System architecture explanations

**Note**: The demo is standalone and can be opened directly in a browser as a static file, or served locally.

## Architecture Notes

- **dashboard.html**: Server-driven via `/api/` endpoints in `wheeler_ui.py` (store, recall, forget, sleep endpoints)
- **demo.html**: Standalone static HTML; includes embedded JavaScript for educational visualization
- Both share design system CSS (dark theme, violet/cyan/magenta accents)

## Future Work

- Merge dashboard and demo into a unified, tabbed interface
- Add real-time CA evolution visualization with WebGL
- Export memory snapshots and attractor correlation matrices
