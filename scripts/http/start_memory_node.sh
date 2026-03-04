#!/bin/bash
# 启动 HNSW 向量存储引擎 Memory Node 的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LOG_DIR="$SCRIPT_DIR/logs"

# 默认配置
MEMORY_SERVER="localhost"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "$LOG_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HNSW 向量存储引擎 Memory Node 启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查是否已经编译
if [ ! -f "$BUILD_DIR/shine" ]; then
    echo -e "${RED}错误: 未找到可执行文件 $BUILD_DIR/shine${NC}"
    echo -e "${YELLOW}请先编译项目:${NC}"
    echo "  cd $PROJECT_DIR"
    echo "  mkdir -p build && cd build"
    echo "  cmake -DCMAKE_BUILD_TYPE=Release .."
    echo "  make -j\$(nproc)"
    exit 1
fi

# 检查 huge pages
HUGE_PAGES=$(cat /proc/sys/vm/nr_hugepages)
echo -e "${YELLOW}当前 Huge Pages: $HUGE_PAGES${NC}"
if [ "$HUGE_PAGES" -lt 1000 ]; then
    echo -e "${RED}警告: Huge Pages 数量可能不足${NC}"
    echo -e "${YELLOW}建议设置: sudo echo 32768 > /proc/sys/vm/nr_hugepages${NC}"
fi

# 检查并杀掉旧的 memory server 进程
echo ""
echo -e "${YELLOW}检查并清理旧的 Memory Server 进程...${NC}"
pkill -9 -f "--is-server" 2>/dev/null || true
sleep 1

# 启动 memory server
echo ""
echo -e "${GREEN}启动 Memory Server${NC}"
echo "命令: $BUILD_DIR/shine --is-server --servers $MEMORY_SERVER"
echo "日志: $LOG_DIR/memory_server.log"

$BUILD_DIR/shine --is-server --servers $MEMORY_SERVER > "$LOG_DIR/memory_server.log" 2>&1 &
MEMORY_PID=$!
echo -e "${GREEN}Memory Server 已启动 (PID: $MEMORY_PID)${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Memory Node 启动完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "进程信息:"
echo "  Memory Server PID: $MEMORY_PID"
echo ""
echo "日志文件:"
echo "  Memory Server: $LOG_DIR/memory_server.log"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_DIR/memory_server.log"
echo ""
echo "停止服务:"
echo "  pkill -9 -f \"--is-server\""
echo ""

# 保存 PID 到文件
echo "$MEMORY_PID" > "$LOG_DIR/memory_server.pid"
