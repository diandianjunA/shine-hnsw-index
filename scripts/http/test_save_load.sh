#!/bin/bash
# 测试 HNSW 向量存储引擎的保存和加载索引功能

HOST="localhost"
PORT="8080"

echo "=== HNSW 向量存储引擎 保存/加载索引 测试 ==="
echo ""

# 健康检查
echo "1. 健康检查"
curl -s "http://${HOST}:${PORT}/health" | python3 -m json.tool
echo ""

# 插入测试向量
echo "2. 插入测试向量"
for i in 1 2 3; do
    VECTOR=$(python3 -c "import random; print(','.join([str(random.random()) for _ in range(10)]))")
    INSERT_PAYLOAD="{\"vector\": [${VECTOR}], \"id\": ${i}}"
    echo "  插入向量 id=${i}: $(echo ${INSERT_PAYLOAD} | head -c 50)..."
    curl -s -X POST -H "Content-Type: application/json" -d "${INSERT_PAYLOAD}" \
        "http://${HOST}:${PORT}/insert" | python3 -m json.tool
done
echo ""

# 保存索引
echo "3. 保存索引"
SAVE_PAYLOAD='{"path": ""}'
curl -s -X POST -H "Content-Type: application/json" -d "${SAVE_PAYLOAD}" \
    "http://${HOST}:${PORT}/save" | python3 -m json.tool
echo ""

# 插入更多向量（测试加载后状态）
echo "4. 插入更多向量 (测试加载后状态)"
for i in 10 11 12; do
    VECTOR=$(python3 -c "import random; print(','.join([str(random.random()) for _ in range(10)]))")
    INSERT_PAYLOAD="{\"vector\": [${VECTOR}], \"id\": ${i}}"
    echo "  插入向量 id=${i}..."
    curl -s -X POST -H "Content-Type: application/json" -d "${INSERT_PAYLOAD}" \
        "http://${HOST}:${PORT}/insert" | python3 -m json.tool
done
echo ""

# 加载索引
echo "5. 加载索引"
LOAD_PAYLOAD='{"path": ""}'
curl -s -X POST -H "Content-Type: application/json" -d "${LOAD_PAYLOAD}" \
    "http://${HOST}:${PORT}/load" | python3 -m json.tool
echo ""

# 查询测试
echo "6. 查询测试 (加载后)"
QUERY_PAYLOAD='{"vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "k": 5, "ef_search": 128}'
curl -s -X POST -H "Content-Type: application/json" -d "${QUERY_PAYLOAD}" \
    "http://${HOST}:${PORT}/query" | python3 -m json.tool
echo ""

echo "=== 测试完成 ==="
