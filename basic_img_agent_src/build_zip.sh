#!/usr/bin/env bash
set -euo pipefail

# Builds a self-contained deployment.zip for AWS Bedrock AgentCore upload.
# All pip dependencies are bundled at the root alongside my_agent.py.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build"
ZIP_NAME="deployment.zip"

echo "==> Cleaning previous build..."
rm -rf "$BUILD_DIR" "$SCRIPT_DIR/$ZIP_NAME"
mkdir -p "$BUILD_DIR"

echo "==> Installing dependencies into build directory..."
pip install \
  --target "$BUILD_DIR" \
  --platform manylinux2014_aarch64 \
  --only-binary=:all: \
  --python-version 3.12 \
  -r "$SCRIPT_DIR/requirements.txt"

echo "==> Copying agent code..."
cp "$SCRIPT_DIR/my_agent.py" "$BUILD_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$BUILD_DIR/"

echo "==> Creating $ZIP_NAME..."
cd "$BUILD_DIR"
zip -r "$SCRIPT_DIR/$ZIP_NAME" . -x '*.pyc' '__pycache__/*'

echo "==> Done! Upload this file to the AWS console:"
echo "    $SCRIPT_DIR/$ZIP_NAME"
echo ""
ls -lh "$SCRIPT_DIR/$ZIP_NAME"
