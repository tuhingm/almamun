#!/bin/sh
# Finish one new OES model after its engine code, registry entry and guide .md are written.
# usage: sh tools/publish_oes_model.sh <section> <guide-slug> <model-id>      e.g.  sh tools/publish_oes_model.sh micro ordered-logit ordered
set -e
cd "$(dirname "$0")/.."
python3 tools/bayes_markov/build_guides.py "$1" "$2"
python3 tools/bayes_markov/build_hubs.py "$1"
python3 tools/oes_sync.py --local --only="$3"
python3 tools/oes_counts.py
python3 tools/author_note.py
python3 tools/site_polish.py
python3 tools/share_buttons.py
FOLDER=$(python3 -c "import sys; sys.path.insert(0,'tools/bayes_markov'); from build_guides import SECTIONS; print(SECTIONS['$1']['folder'])")
OG_ONLY="assets/og/$FOLDER" CHROME=${CHROME:-/usr/bin/google-chrome} python3 tools/make_og_images.py
python3 tools/og_tags.py
python3 tools/make_sitemap.py
