#!/bin/bash
# 简单的 HTTP API 测试脚本

HOST="localhost"
PORT="8080"

echo "=== HNSW 向量存储引擎 HTTP API 测试 ==="
echo ""

# 健康检查
echo "1. 健康检查"
curl -s "http://${HOST}:${PORT}/health" | python3 -m json.tool
echo ""

# 获取系统信息
echo "2. 系统信息"
curl -s "http://${HOST}:${PORT}/info" | python3 -m json.tool
echo ""

# 插入向量测试
echo "3. 插入向量测试"
INSERT_PAYLOAD='{"vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
curl -s -X POST -H "Content-Type: application/json" -d "$INSERT_PAYLOAD" \
    "http://${HOST}:${PORT}/insert" | python3 -m json.tool
echo ""

# 查询向量测试
echo "4. 查询向量测试"
QUERY_PAYLOAD='{"vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "k": 5, "ef_search": 128}'
curl -s -X POST -H "Content-Type: application/json" -d "$QUERY_PAYLOAD" \
    "http://${HOST}:${PORT}/query" | python3 -m json.tool
echo ""

echo "=== 测试完成 ==="