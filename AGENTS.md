# AGENTS.md

## Project overview
This repository contains a Qt Widgets C++ desktop application for demonstrating a drone detection model.

Current stack:
- Qt 6
- C++
- qmake
- OpenCV

The project is intended to evolve from a minimal desktop demo into a structured application that can:
- load and display images
- load and display video
- later support camera input
- run detection on frames
- draw detection results in the UI
- later integrate a real trained model backend

---

## Main development goals
1. Keep the application buildable at every step.
2. Keep architecture modular from the beginning.
3. Separate UI, control flow, data structures, and detection logic.
4. Avoid large rewrites when small incremental changes are enough.
5. Prefer vertical slices: each task should end in a working, testable state.

---

## Architecture rules
Use a layered structure and keep responsibilities separated.

Suggested repository structure:

- `main.cpp` — application entry point
- `mainwindow.h/.cpp/.ui` — UI layer only
- `controller/` — orchestration and application flow
- `detection/` — detector interfaces, detection data structures, detector implementations
- `sources/` — image/video/camera frame source abstractions
- `utils/` — helper functions such as OpenCV ↔ Qt conversions and drawing utilities

### Responsibility boundaries
- `MainWindow` should contain UI wiring only.
- Business logic should not be placed directly into `MainWindow` unless absolutely necessary.
- Detection pipeline coordination should go through a controller class.
- Detector-specific logic should be isolated behind interfaces where reasonable.
- OpenCV image handling should remain outside the UI layer as much as possible.

---

## Coding constraints
- Use C++ and Qt Widgets only.
- Keep qmake project structure unless explicitly asked to migrate.
- Keep OpenCV integration compatible with Qt MSVC build.
- Do not introduce Python dependency unless explicitly requested.
- Do not introduce heavy frameworks unless explicitly requested.
- Prefer adding new files over rewriting existing working code.
- Keep naming straightforward and predictable.
- Avoid premature overengineering, but preserve extension points.

---

## Build expectations
The project should remain compilable in Qt Creator on Windows with:
- Qt 6
- MSVC 2022 64-bit
- OpenCV prebuilt binaries

When modifying the `.pro` file:
- preserve existing working configuration
- add new files carefully
- do not remove required Qt modules
- keep OpenCV include/lib settings intact unless intentionally updating them

---

## UI expectations
At early stages, keep the UI simple and practical.

Preferred early UI features:
- open image button
- open video button
- image/frame display area
- status text or status bar
- run detection button when appropriate

Do not add complex styling unless explicitly requested.

---

## Detection expectations
The detection subsystem should be developed incrementally:

1. Start with a stub detector that returns fake detections.
2. Add rendering of bounding boxes.
3. Add image loading and processing flow.
4. Add video frame processing flow.
5. Only then integrate the real inference backend.

This allows the application architecture to be tested before real model integration.

---

## Preferred implementation style
- Small, focused commits
- Minimal invasive changes
- Clear class boundaries
- Readable code over clever code
- Add comments only where they genuinely help
- Preserve existing behavior unless the task explicitly changes it

---

## What to avoid
- Do not move all logic into one file.
- Do not rewrite the entire project for a small feature.
- Do not mix temporary experiments into production code.
- Do not break the build to partially implement a feature.
- Do not replace the architecture with a shortcut unless explicitly asked.

---

## First milestone expectations
A good first milestone is:
- open an image from disk
- show it in the main window
- pass it through a controller
- run a stub detector
- draw at least one fake bounding box
- keep the project buildable and easy to extend