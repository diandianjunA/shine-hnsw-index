#!/bin/bash
# 启动 HNSW 向量存储引擎 Compute Node 的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LOG_DIR="$SCRIPT_DIR/logs"

# 默认配置
HTTP_PORT=8080
NUM_THREADS=4
EF_SEARCH=128
K=10
# MEMORY_SERVER="localhost"
MEMORY_SERVER="cluster1"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "$LOG_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HNSW 向量存储引擎 Compute Node 启动脚本${NC}"
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

# 检查并杀掉旧的 compute node 进程
echo ""
echo -e "${YELLOW}检查并清理旧的 Compute Node 进程...${NC}"
pkill -9 -f "--enable-http" 2>/dev/null || true
sleep 1

# 启动 compute node (HTTP server)
echo ""
echo -e "${GREEN}启动 Compute Node (HTTP Server)${NC}"
echo "命令: $BUILD_DIR/shine --enable-http --http-host 0.0.0.0 --http-port $HTTP_PORT --servers $MEMORY_SERVER --initiator -t $NUM_THREADS --ef-search $EF_SEARCH -k $K --cache"
echo "日志: $LOG_DIR/http_server.log"

$BUILD_DIR/shine --enable-http --http-host 0.0.0.0 --http-port $HTTP_PORT --servers $MEMORY_SERVER --initiator -t $NUM_THREADS --ef-search $EF_SEARCH -k $K --cache > "$LOG_DIR/http_server.log" 2>&1 &
HTTP_PID=$!
echo -e "${GREEN}HTTP Server 已启动 (PID: $HTTP_PID)${NC}"

sleep 5

# 测试连接
echo ""
echo -e "${GREEN}测试 HTTP API${NC}"
echo "测试健康检查..."

for i in {1..10}; do
    HEALTH=$(curl -s http://localhost:$HTTP_PORT/health 2>&1 || echo "failed")
    if echo "$HEALTH" | grep -q "success"; then
        echo -e "${GREEN}✓ HTTP API 正常运行${NC}"
        echo "响应: $HEALTH"
        break
    else
        if [ $i -eq 10 ]; then
            echo -e "${RED}✗ HTTP API 未响应${NC}"
            echo ""
            echo "查看日志:"
            echo "  HTTP Server: tail -f $LOG_DIR/http_server.log"
            echo ""
            echo "检查进程:"
            echo "  ps aux | grep shine"
        else
            echo -e "${YELLOW}等待服务启动... ($i/10)${NC}"
            sleep 2
        fi
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Compute Node 启动完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "进程信息:"
echo "  HTTP Server PID: $HTTP_PID"
echo ""
echo "HTTP API 地址: http://localhost:$HTTP_PORT"
echo ""
echo "日志文件:"
echo "  HTTP Server: $LOG_DIR/http_server.log"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_DIR/http_server.log"
echo ""
echo "停止服务:"
echo "  pkill -9 -f \"--enable-http\""
echo ""
echo "测试命令:"
echo "  cd $SCRIPT_DIR"
echo "  python3 test_api.py --port $HTTP_PORT"
echo ""

# 保存 PID 到文件
echo "$HTTP_PID" > "$LOG_DIR/http_server.pid"
