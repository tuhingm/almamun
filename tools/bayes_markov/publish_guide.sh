#!/bin/sh
# Build one Bayesian or Markov guide and run the site-wide finishing pass.  usage: sh tools/bayes_markov/publish_guide.sh bayesian|markov <slug>
set -e
cd "$(dirname "$0")/../.."
python3 tools/bayes_markov/build_guides.py "$1" "$2"
python3 tools/bayes_markov/add_guide.py "$1"
python3 tools/author_note.py
python3 tools/site_polish.py
python3 tools/share_buttons.py
FOLDER=$(python3 -c "import sys; sys.path.insert(0,'tools/bayes_markov'); from build_guides import SECTIONS; print(SECTIONS['$1']['folder'])")
OG_ONLY="assets/og/$FOLDER/$2" python3 tools/make_og_images.py
python3 tools/og_tags.py
python3 tools/make_sitemap.py
