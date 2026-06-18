#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# ASTER Setup & Launch Script
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────────

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   ASTER – Adaptive Smart Traffic Event Response          ║${NC}"
echo -e "${BOLD}${CYAN}║   Bengaluru Traffic Intelligence System                  ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3.10+ required but not found.${NC}"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}[✓]${NC} Python ${PYVER} found"

# Install dependencies
echo -e "\n${CYAN}[1/3] Installing dependencies...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}[✓]${NC} Dependencies installed"

# Check data file
if [ ! -f "data/bengaluru_traffic_events.csv" ]; then
    echo -e "${RED}[ERROR] Data file not found: data/bengaluru_traffic_events.csv${NC}"
    exit 1
fi
echo -e "${GREEN}[✓]${NC} Dataset found"

# Train if model doesn't exist
if [ ! -f "models/gb_main.pkl" ]; then
    echo -e "\n${CYAN}[2/3] Training model (first run only)...${NC}"
    python3 train.py
    echo -e "${GREEN}[✓]${NC} Model trained and saved"
else
    echo -e "${GREEN}[✓]${NC} Trained model found — skipping training"
fi

# Verify assets
echo -e "\n${CYAN}[3/3] Verifying EDA assets...${NC}"
python3 eda.py -q 2>/dev/null || python3 eda.py
echo -e "${GREEN}[✓]${NC} Assets ready"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  🚦 ASTER is ready!${NC}"
echo ""
echo -e "  ${YELLOW}Option A — Streamlit app (recommended):${NC}"
echo -e "  ${BOLD}streamlit run app/aster_app.py${NC}"
echo ""
echo -e "  ${YELLOW}Option B — Standalone HTML demo (no server needed):${NC}"
echo -e "  ${BOLD}open app/index.html${NC}  (or double-click the file)"
echo ""
echo -e "  ${YELLOW}Option C — CLI prediction test:${NC}"
echo -e "  ${BOLD}python3 predict.py --cause accident --corridor \"Mysore Road\" --hour 8 --closure${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
