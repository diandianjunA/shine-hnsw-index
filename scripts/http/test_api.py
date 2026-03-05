#!/usr/bin/env python3
"""
测试脚本 - 用于测试 HNSW 向量存储引擎的 HTTP API
"""

import requests
import json
import random
import time
import argparse
from typing import List, Dict, Any

class HnswClient:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}"
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def insert_vector(self, vector: List[float], vector_id: int = None) -> Dict[str, Any]:
        """插入向量"""
        payload = {"vector": vector}
        if vector_id is not None:
            payload["id"] = vector_id
        
        try:
            response = requests.post(
                f"{self.base_url}/insert",
                json=payload,
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def query_vector(self, vector: List[float], k: int = 10, ef_search: int = 128) -> Dict[str, Any]:
        """查询向量"""
        payload = {
            "vector": vector,
            "k": k,
            "ef_search": ef_search
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def save_index(self, path: str = "") -> Dict[str, Any]:
        """保存索引到文件"""
        payload = {"path": path}
        
        try:
            response = requests.post(
                f"{self.base_url}/save",
                json=payload,
                timeout=60
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def load_index(self, path: str = "") -> Dict[str, Any]:
        """从文件加载索引"""
        payload = {"path": path}
        
        try:
            response = requests.post(
                f"{self.base_url}/load",
                json=payload,
                timeout=60
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}


def generate_random_vector(dim: int) -> List[float]:
    """生成随机向量"""
    return [random.random() for _ in range(dim)]


def test_basic_insert_query(client: HnswClient, dim: int = 128, num_vectors: int = 100):
    """测试基本的插入和查询功能"""
    print(f"\n=== 测试基本插入和查询 (维度={dim}, 向量数={num_vectors}) ===")
    
    # 插入向量
    print(f"\n1. 插入 {num_vectors} 个向量...")
    inserted_ids = []
    vectors = []
    
    for i in range(num_vectors):
        vector = generate_random_vector(dim)
        vectors.append(vector)
        
        result = client.insert_vector(vector, vector_id=i)
        if result.get("success"):
            inserted_ids.append(result.get("id", i))
            if (i + 1) % 10 == 0:
                print(f"  已插入 {i + 1}/{num_vectors} 个向量")
        else:
            print(f"  插入失败 (id={i}): {result.get('error')}")
    
    print(f"\n成功插入 {len(inserted_ids)} 个向量")
    
    # 查询测试
    print(f"\n2. 查询测试...")
    if vectors:
        # 查询第一个向量
        query_vector = vectors[0]
        result = client.query_vector(query_vector, k=5)
        
        if result.get("success"):
            print(f"  查询成功!")
            print(f"  返回结果数: {len(result.get('results', []))}")
            print(f"  结果: {result.get('results', [])}")
        else:
            print(f"  查询失败: {result.get('error')}")


def test_performance(client: HnswClient, dim: int = 128, num_vectors: int = 1000):
    """性能测试"""
    print(f"\n=== 性能测试 (维度={dim}, 向量数={num_vectors}) ===")
    
    # 批量插入
    print(f"\n1. 批量插入 {num_vectors} 个向量...")
    start_time = time.time()
    
    for i in range(num_vectors):
        vector = generate_random_vector(dim)
        client.insert_vector(vector, vector_id=i)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  已插入 {i + 1}/{num_vectors} 个向量, 耗时: {elapsed:.2f}s")
    
    insert_time = time.time() - start_time
    print(f"\n插入完成! 总耗时: {insert_time:.2f}s, 平均: {insert_time/num_vectors*1000:.2f}ms/个")
    
    # 批量查询
    print(f"\n2. 批量查询 100 次...")
    start_time = time.time()
    
    for i in range(100):
        vector = generate_random_vector(dim)
        client.query_vector(vector, k=10)
    
    query_time = time.time() - start_time
    print(f"\n查询完成! 总耗时: {query_time:.2f}s, 平均: {query_time/100*1000:.2f}ms/次")


def main():
    parser = argparse.ArgumentParser(description="HNSW 向量存储引擎 HTTP API 测试")
    parser.add_argument("--host", default="localhost", help="服务器地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口 (默认: 8080)")
    parser.add_argument("--dim", type=int, default=128, help="向量维度 (默认: 128)")
    parser.add_argument("--num-vectors", type=int, default=100, help="测试向量数 (默认: 100)")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")
    
    args = parser.parse_args()
    
    client = HnswClient(args.host, args.port)
    
    # 健康检查
    print("=== 健康检查 ===")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    if not health.get("success"):
        print("\n错误: 无法连接到服务器!")
        return
    
    # 获取系统信息
    print("\n=== 系统信息 ===")
    info = client.get_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # 运行基本测试
    test_basic_insert_query(client, args.dim, args.num_vectors)
    
    # 运行性能测试
    if args.performance:
        test_performance(client, args.dim, num_vectors=1000)
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()