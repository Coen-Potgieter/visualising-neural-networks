#!/bin/bash

set -e # Exit the script if there is some error

# Define Some Variables for Re-Use
TARGET_SCRIPT="main.py"  
VENV_DIR="env"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
RED='\033[0;31m'
NC='\033[0m' 

# Ask For python3
PYTHON_CMD="python"
echo -e "${YELLOW}Would you like to use python3? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    PYTHON_CMD="python3"
fi

# Ask which script to run
echo -e "${YELLOW}Which script would you like to run?\n1) main.py\n2) 3d-demo.py\n3) ml-img.py\nEnter Either 1, 2 or 3: ${NC}"
read -r response
case "$response" in
    1) TARGET_SCRIPT="main.py" ;;
    2) TARGET_SCRIPT="3d_demo.py" ;;
    3) TARGET_SCRIPT="ml-img.py" ;;
    *) echo "Invalid input. Please enter 1, 2, or 3."; exit 1 ;;
esac

echo -e "${BLUE}Starting setup with ${PYTHON_CMD}...${NC}"

echo -e "${YELLOW}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv "$VENV_DIR"

echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

echo -e "${YELLOW}Running the script: $TARGET_SCRIPT${NC}"
$PYTHON_CMD "$TARGET_SCRIPT"

echo -e "${YELLOW}Deactivating virtual environment...${NC}"
deactivate

echo -e "${YELLOW}Cleaning...${NC}"
rm -rf "$VENV_DIR"/

echo -e "${GREEN}Demo complete${NC}"




