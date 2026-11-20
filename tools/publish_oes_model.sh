#!/bin/sh
# Finish one new OES model after its engine code, registry entry and guide .md are written.
# usage: sh tools/publish_oes_model.sh <section> <guide-slug> <model-id>      e.g.  sh tools/publish_oes_model.sh micro ordered-logit ordered
set -e
cd "$(dirname "$0")/.."
python3 tools/bayes_markov/build_guides.py "$1" "$2"
# the time-series and panel-data guides (sections lp, plp) have hand-built hubs; the others are generated
if python3 -c "import sys; sys.path.insert(0,'tools/bayes_markov'); from build_hubs import HUBS; sys.exit(0 if '$1' in HUBS else 1)"; then python3 tools/bayes_markov/build_hubs.py "$1"; fi
# the Causal Inference Guide has its own tool page: register the new guide in it
if [ "$1" = "causal" ]; then python3 tools/causal_selector/add_causal_model.py "$2"; fi
# the Machine Learning Guide has its own tool page: list the new guide there
if [ "$1" = "ml" ]; then python3 tools/ml_selector/add_ml_model.py "$2"; fi
python3 tools/oes_sync.py --local --only="$3"
python3 tools/oes_counts.py
python3 tools/author_note.py
python3 tools/site_polish.py
python3 tools/share_buttons.py
FOLDER=$(python3 -c "import sys; sys.path.insert(0,'tools/bayes_markov'); from build_guides import SECTIONS; print(SECTIONS['$1']['folder'])")
OG_ONLY="assets/og/$FOLDER" CHROME=${CHROME:-/usr/bin/google-chrome} python3 tools/make_og_images.py
python3 tools/og_tags.py
python3 tools/make_sitemap.py
