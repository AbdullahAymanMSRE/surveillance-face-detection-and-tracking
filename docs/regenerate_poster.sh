#!/usr/bin/env bash
#
# Regenerate Capstone_Poster.pdf and Capstone_Poster.png from the HTML.
#
#   ./regenerate_poster.sh          # PNG at 80 dpi (default)
#   ./regenerate_poster.sh 150      # PNG at 150 dpi (higher-res proof)
#
# Requires: Google Chrome + python3. The script installs PyMuPDF if missing.

set -euo pipefail
cd "$(dirname "$0")"

HTML="Capstone_Poster_Template.html"
PDF="Capstone_Poster.pdf"
PNG="Capstone_Poster.png"
DPI="${1:-80}"

# --- locate Chrome (macOS, then common Linux names) ---
CHROME=""
for c in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium" \
  "$(command -v google-chrome 2>/dev/null || true)" \
  "$(command -v chromium 2>/dev/null || true)" \
  "$(command -v chromium-browser 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CHROME="$c"; break; fi
done
if [ -z "$CHROME" ]; then
  echo "ERROR: Google Chrome / Chromium not found." >&2
  exit 1
fi

URL="file://$PWD/$HTML"

echo "→ Rendering PDF (85×200 cm)…"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/$PDF" "$URL" 2>/dev/null

echo "→ Rendering PNG (${DPI} dpi)…"
if python3 - "$PDF" "$PNG" "$DPI" <<'PY'
import sys, subprocess
try:
    import fitz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pymupdf"])
    import fitz
pdf, png, dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = fitz.open(pdf); p = d[0]
w = p.rect.width / 72 * 25.4 / 10
h = p.rect.height / 72 * 25.4 / 10
pix = p.get_pixmap(dpi=dpi); pix.save(png)
print(f"  {d.page_count} page(s) · {w:.1f}×{h:.1f} cm · PNG {pix.width}×{pix.height}px")
if d.page_count > 1:
    print("  ⚠️  WARNING: content overflows to %d pages — it no longer fits 85×200cm." % d.page_count)
    print("     Trim a little (spacing/figure/text) and re-run, or ask Claude to fix it.")
    sys.exit(2)
PY
then
  echo "Done → $PDF, $PNG"
else
  echo "Done (SEE WARNING ABOVE) → $PDF, $PNG"
fi
