#!/usr/bin/env python3
"""
测试脚本 - 用于测试 HNSW 向量存储引擎的保存和加载索引功能

测试流程:
1. 插入多个向量
2. 保存索引到文件
3. 验证保存成功
4. 重新插入一些新向量（测试加载后是否保留原有数据或清空）
5. 加载索引
6. 验证加载成功
7. 测试查询功能
"""

import requests
import json
import random
import time
import argparse
import os
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


def test_save_index(client: HnswClient, dim: int = 128, num_vectors: int = 50):
    """测试保存索引功能"""
    print(f"\n=== 测试保存索引 (维度={dim}, 向量数={num_vectors}) ===")
    
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
    
    # 保存索引
    print(f"\n2. 保存索引...")
    save_result = client.save_index("")
    
    if save_result.get("success"):
        print(f"  保存成功!")
        print(f"  保存路径: {save_result.get('path', 'N/A')}")
        return save_result.get("path")
    else:
        print(f"  保存失败: {save_result.get('error')}")
        return None


def test_load_index(client: HnswClient, dim: int = 128, num_new_vectors: int = 10):
    """测试加载索引功能"""
    print(f"\n=== 测试加载索引 (新向量数={num_new_vectors}) ===")
    
    # 先插入一些新向量（模拟重启后新增的数据）
    print(f"\n1. 插入 {num_new_vectors} 个新向量...")
    for i in range(num_new_vectors):
        vector = generate_random_vector(dim)
        result = client.insert_vector(vector, vector_id=1000 + i)
        if not result.get("success"):
            print(f"  插入失败 (id={1000+i}): {result.get('error')}")
    print(f"  完成插入 {num_new_vectors} 个新向量")
    
    # 加载索引
    print(f"\n2. 加载索引...")
    load_result = client.load_index("")
    
    if load_result.get("success"):
        print(f"  加载成功!")
        print(f"  加载路径: {load_result.get('path', 'N/A')}")
        return True
    else:
        print(f"  加载失败: {load_result.get('error')}")
        return False


def test_query_after_load(client: HnswClient, dim: int = 128):
    """测试加载后查询功能"""
    print(f"\n=== 测试加载后查询功能 ===")
    
    # 查询测试
    print(f"\n1. 执行查询测试...")
    query_vector = generate_random_vector(dim)
    result = client.query_vector(query_vector, k=5)
    
    if result.get("success"):
        print(f"  查询成功!")
        print(f"  返回结果数: {len(result.get('results', []))}")
        print(f"  结果: {result.get('results', [])}")
        return True
    else:
        print(f"  查询失败: {result.get('error')}")
        return False


def test_full_workflow(client: HnswClient, dim: int = 128, num_vectors: int = 50):
    """完整的工作流测试: 插入 -> 保存 -> 插入新数据 -> 加载 -> 查询"""
    print(f"\n{'='*60}")
    print(f"完整工作流测试: 插入 -> 保存 -> 插入新数据 -> 加载 -> 查询")
    print(f"{'='*60}")
    
    # Step 1: 插入初始向量
    print(f"\n[Step 1] 插入 {num_vectors} 个初始向量...")
    for i in range(num_vectors):
        vector = generate_random_vector(dim)
        result = client.insert_vector(vector, vector_id=i)
        if not result.get("success"):
            print(f"  插入失败 (id={i}): {result.get('error')}")
    print(f"  完成插入 {num_vectors} 个初始向量")
    
    # Step 2: 保存索引
    print(f"\n[Step 2] 保存索引...")
    save_result = client.save_index("")
    if not save_result.get("success"):
        print(f"  保存失败: {save_result.get('error')}")
        return False
    save_path = save_result.get("path")
    print(f"  保存成功: {save_path}")
    
    # 验证文件存在
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"  文件大小: {file_size} bytes")
    else:
        print(f"  警告: 保存的文件不存在于 {save_path}")
    
    # Step 3: 插入新向量
    print(f"\n[Step 3] 插入 {num_vectors} 个新向量...")
    for i in range(num_vectors, num_vectors * 2):
        vector = generate_random_vector(dim)
        result = client.insert_vector(vector, vector_id=i)
        if not result.get("success"):
            print(f"  插入失败 (id={i}): {result.get('error')}")
    print(f"  完成插入 {num_vectors} 个新向量")
    
    # Step 4: 加载索引
    print(f"\n[Step 4] 加载索引...")
    load_result = client.load_index("")
    if not load_result.get("success"):
        print(f"  加载失败: {load_result.get('error')}")
        return False
    print(f"  加载成功: {load_result.get('path')}")
    
    # Step 5: 查询测试
    print(f"\n[Step 5] 执行查询测试...")
    query_vector = generate_random_vector(dim)
    result = client.query_vector(query_vector, k=10)
    
    if result.get("success"):
        print(f"  查询成功!")
        print(f"  返回结果数: {len(result.get('results', []))}")
    else:
        print(f"  查询失败: {result.get('error')}")
        return False
    
    print(f"\n{'='*60}")
    print(f"完整工作流测试通过!")
    print(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="HNSW 向量存储引擎保存/加载索引测试")
    parser.add_argument("--host", default="localhost", help="服务器地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口 (默认: 8080)")
    parser.add_argument("--dim", type=int, default=128, help="向量维度 (默认: 128)")
    parser.add_argument("--num-vectors", type=int, default=50, help="测试向量数 (默认: 50)")
    parser.add_argument("--skip-insert", action="store_true", help="跳过插入步骤(假设已有数据)")
    
    args = parser.parse_args()
    
    client = HnswClient(args.host, args.port)
    
    # 健康检查
    print("=== 健康检查 ===")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    if not health.get("success"):
        print("\n错误: 无法连接到服务器!")
        return
    
    # 运行完整工作流测试
    success = test_full_workflow(client, args.dim, args.num_vectors)
    
    if success:
        print("\n=== 所有测试通过 ===")
    else:
        print("\n=== 测试失败 ===")


if __name__ == "__main__":
    main()
