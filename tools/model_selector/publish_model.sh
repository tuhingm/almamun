#!/bin/sh
# Build one extra Time-Series guide and run the site-wide finishing pass.  usage: sh tools/model_selector/publish_model.sh <slug>
# (preview images need Playwright with Chrome; set CHROME and OG_FONT_CSS as described in tools/make_og_images.py)
set -e
cd "$(dirname "$0")/../.."
python3 tools/bayes_markov/build_guides.py lp "$1"
python3 tools/model_selector/add_model.py "$1"
python3 tools/author_note.py
python3 tools/site_polish.py
python3 tools/share_buttons.py
OG_ONLY="assets/og/time-series-guide-$1" python3 tools/make_og_images.py
python3 tools/og_tags.py
python3 tools/make_sitemap.py
