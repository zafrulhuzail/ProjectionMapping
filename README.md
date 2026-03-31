# Beamer Projection Mapping

![Projection mapping demo](docs/demo-pm.jfif)

A projection-mapping workspace combining:

- a baseline ArUco / homography pipeline (`Js_projection_mapping/`)
- OpenCV card-recognition experiments (`OpenCV_tuto/`)
- newer runnable entry points in `src/`

## Structure

- `src/` — current runnable scripts for detection + projection flows
- `Js_projection_mapping/` — original 3-stage calibration / homography / tracking pipeline
- `OpenCV_tuto/` — older OpenCV experiments and reference assets
- `docs/` — project notes and conceptual explanation
- `runs/` — generated model output / detections (ignored)

## Recommended starting points

### Original projection-mapping pipeline
1. `Js_projection_mapping/01_intrinsic_calibration/`
2. `Js_projection_mapping/02_homogrphic_transform/`
3. `Js_projection_mapping/03_aruco_tracking/`

### Newer runnable scripts
- `src/run_projection_with_yolo.py`
- `src/run_projection_with_card_header_cv.py`
- `src/run_projection_with_header_ocr.py`
- `src/run_model.py`

## Notes

- `Anleitung_Verstandnis.md` and `notes.md` were moved into `docs/` for clarity.
- `.venv/`, `__pycache__/`, and `runs/` are ignored so the repo stays clean.
- The folder name `02_homogrphic_transform` is kept as-is for compatibility with existing paths.
