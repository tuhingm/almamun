#!/bin/sh
# Build one extra Causal guide and run the site-wide finishing pass.  usage: sh tools/causal_selector/publish_causal_model.sh <slug>
set -e
cd "$(dirname "$0")/../.."
python3 tools/causal_selector/build_causal_guides.py "$1"
python3 tools/causal_selector/add_causal_model.py "$1"
python3 tools/author_note.py
python3 tools/site_polish.py
python3 tools/share_buttons.py
OG_ONLY="assets/og/causal-inference-guide/$1" python3 tools/make_og_images.py
python3 tools/og_tags.py
python3 tools/make_sitemap.py
