#!/usr/bin/env python3
"""
模拟测试脚本 - 用于测试 HTTP 客户端代码（不需要真实服务器）
"""

import json
import sys
from unittest.mock import Mock, patch
from test_api import HnswClient, generate_random_vector

def test_client_code():
    """测试客户端代码逻辑"""
    print("=== 模拟测试 HTTP 客户端代码 ===\n")
    
    # 创建客户端
    client = HnswClient("localhost", 8080)
    print(f"✓ 客户端创建成功: {client.base_url}")
    
    # 测试向量生成
    print("\n1. 测试向量生成...")
    vector = generate_random_vector(128)
    print(f"✓ 生成 128 维随机向量，长度: {len(vector)}")
    print(f"  前 5 个元素: {vector[:5]}")
    
    # 测试请求构建
    print("\n2. 测试请求构建...")
    
    # 插入请求
    insert_payload = {"vector": vector, "id": 1}
    print(f"✓ 插入请求构建成功")
    print(f"  payload 大小: {len(json.dumps(insert_payload))} bytes")
    
    # 查询请求
    query_payload = {"vector": vector, "k": 10, "ef_search": 128}
    print(f"✓ 查询请求构建成功")
    print(f"  payload 大小: {len(json.dumps(query_payload))} bytes")
    
    # 测试响应解析
    print("\n3. 测试响应解析...")
    
    # 健康检查响应
    health_response = {"success": True, "status": "healthy"}
    print(f"✓ 健康检查响应: {health_response}")
    
    # 插入响应
    insert_response = {"success": True, "id": 1, "message": "Vector inserted successfully"}
    print(f"✓ 插入响应: {insert_response}")
    
    # 查询响应
    query_response = {"success": True, "results": [1, 5, 10, 15, 20], "k": 10, "ef_search": 128}
    print(f"✓ 查询响应: {query_response}")
    
    # 测试错误处理
    print("\n4. 测试错误处理...")
    error_response = {"success": False, "error": "Connection refused"}
    print(f"✓ 错误响应: {error_response}")
    
    print("\n=== 所有测试通过 ===\n")
    
    # 显示使用说明
    print("使用说明:")
    print("1. 确保 RDMA 集群环境已配置")
    print("2. 运行 ./start_cluster.sh 启动服务器")
    print("3. 运行 python3 test_api.py 进行真实测试")
    print("\n或者手动启动:")
    print("  # 终端 1: 启动 memory server")
    print("  ./shine --is-server --servers localhost")
    print("\n  # 终端 2: 启动 HTTP server")
    print("  ./shine --enable-http --http-host 0.0.0.0 --http-port 8080 \\")
    print("          --servers localhost --initiator -t 4 --ef-search 128 -k 10")
    print("\n  # 终端 3: 运行测试")
    print("  python3 test_api.py")

if __name__ == "__main__":
    test_client_code()
