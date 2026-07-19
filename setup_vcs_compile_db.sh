#!/usr/bin/env bash
# Generate a combined compile_commands.json (simulation core + firmware) at the
# project root for IDE / clangd autocompletion.
#
# Requires: bear and jq for the C simulation core, and PlatformIO (pio) for the
# firmware half.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- Simulation core (root Makefile) ---
make clean
bear --output core.json -- make -j"$(nproc)"

# --- Firmware (PlatformIO / STM32Duino + SimpleFOC) ---
FW_BOARD="firmware/boards/nucleo-g431rb-simplefoc"
( cd "$FW_BOARD" && pio run -t compiledb )
FW_DB="$FW_BOARD/.pio/build/nucleo_g431rb/compile_commands.json"

# --- Merge ---
jq -s 'add' core.json "$FW_DB" > compile_commands.json
rm -f core.json

echo "Updated compile_commands.json in project root."
