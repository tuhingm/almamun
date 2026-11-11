# Lecture videos

Narrated slide videos ("faceless tutorials") are generated from a script per lecture:
`tools/video/<course>-<NN>/script.py` holds the slides (HTML with MathJax) and the narration for each build step.

Rendering needs Chrome + Playwright, ffmpeg, MathJax and the DM Sans/Manrope fonts (npm), and the
Kokoro text-to-speech model (`pip install kokoro-onnx soundfile`; model files from the kokoro-onnx GitHub releases).
`render.py` writes the MP4 and a WebVTT caption file. Copy both into `teaching/<course>/videos/` named after the
lecture file (e.g. `01-probability-regression.mp4/.vtt`, plus a `.jpg` poster), then run `python3 tools/build_courses.py`:
the lecture page picks up the video automatically.
