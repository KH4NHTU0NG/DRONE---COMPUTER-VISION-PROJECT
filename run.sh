#!/bin/bash
# =========================================================
# SAR System Launcher for macOS (iPhone Continuity Camera)
# =========================================================

set -e
cd "$(dirname "$0")"

# Ưu tiên Python từ pyenv shims (nơi đã cài đặt đầy đủ opencv, torch, ultralytics)
if [ -x "$HOME/.pyenv/shims/python" ]; then
    PYTHON_BIN="$HOME/.pyenv/shims/python"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

echo "🚀 Đang khởi động SAR Human Detection..."
echo "📍 Trình thông dịch: $PYTHON_BIN ($($PYTHON_BIN --version))"
echo "🌐 Web Stream MJPEG: http://localhost:5001"
echo "🛑 Bấm phím 'q' trên cửa sổ video hoặc Ctrl+C để dừng."
echo ""

exec "$PYTHON_BIN" App.py "$@"
