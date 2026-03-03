# HTTP API 测试脚本

这个目录包含了用于测试 HNSW 向量存储引擎 HTTP API 的脚本。

## 文件说明

- `test_api.py` - Python 测试脚本，提供完整的功能测试和性能测试
- `test_api.sh` - Shell 测试脚本，用于快速测试基本 API 功能
- `test_mock.py` - 模拟测试脚本，用于测试客户端代码（不需要真实服务器）
- `start_cluster.sh` - 集群启动脚本，自动启动 memory server 和 HTTP server

## 快速开始

### 方式 1: 使用模拟测试（推荐用于验证客户端代码）

```bash
python3 test_mock.py
```

这个脚本不需要真实的服务器，用于验证客户端代码逻辑是否正确。

### 方式 2: 启动真实集群并测试

#### 步骤 1: 配置环境

确保系统已配置 Huge Pages：
```bash
# 检查当前配置
cat /proc/sys/vm/nr_hugepages

# 如果不足，设置 Huge Pages（需要 root 权限）
sudo echo 32768 > /proc/sys/vm/nr_hugepages
```

#### 步骤 2: 启动集群

**自动启动（推荐）**：
```bash
./start_cluster.sh
```

这个脚本会自动启动：
- Memory Server（存储节点）
- Compute Node with HTTP Server（计算节点 + HTTP 服务器）

**手动启动**：
```bash
# 终端 1: 启动 memory server
cd /home/xjs/experiment/shine-hnsw-index/build
./shine --is-server --servers localhost

# 终端 2: 启动 HTTP server
./shine --enable-http --http-host 0.0.0.0 --http-port 8080 \
        --servers localhost --initiator -t 4 --ef-search 128 -k 10
```

#### 步骤 3: 运行测试

```bash
# Python 测试
python3 test_api.py

# Shell 测试
./test_api.sh
```

## 使用方法

### 1. Python 测试脚本

#### 基本用法
```bash
python3 test_api.py --host localhost --port 8080
```

#### 参数说明
- `--host`: 服务器地址（默认：localhost）
- `--port`: 服务器端口（默认：8080）
- `--dim`: 向量维度（默认：128）
- `--num-vectors`: 测试向量数量（默认：100）
- `--performance`: 运行性能测试（可选）

#### 示例
```bash
# 基本测试
python3 test_api.py

# 运行性能测试
python3 test_api.py --performance --num-vectors 1000

# 指定服务器地址和端口
python3 test_api.py --host 192.168.1.100 --port 8888
```

### 2. Shell 测试脚本

#### 基本用法
```bash
./test_api.sh
```

#### 修改服务器地址和端口
编辑脚本中的以下变量：
```bash
HOST="localhost"
PORT="8080"
```

## API 端点说明

### 1. 健康检查
- **端点**: `GET /health`
- **说明**: 检查服务器是否正常运行
- **响应示例**:
```json
{
  "success": true,
  "status": "healthy"
}
```

### 2. 系统信息
- **端点**: `GET /info`
- **说明**: 获取系统信息
- **响应示例**:
```json
{
  "success": true,
  "version": "1.0.0",
  "status": "running"
}
```

### 3. 插入向量
- **端点**: `POST /insert`
- **请求体**:
```json
{
  "id": 123,
  "vector": [0.1, 0.2, 0.3, ..., 0.128]
}
```
- **说明**: 
  - `id` 可选，不提供时自动生成
  - `vector` 必需，向量数据
- **响应示例**:
```json
{
  "success": true,
  "id": 123,
  "message": "Vector inserted successfully"
}
```

### 4. 查询向量
- **端点**: `POST /query`
- **请求体**:
```json
{
  "vector": [0.1, 0.2, 0.3, ..., 0.128],
  "k": 10,
  "ef_search": 128
}
```
- **说明**:
  - `vector` 必需，查询向量
  - `k` 必需，返回最近邻居数量
  - `ef_search` 可选，搜索时的 beam 宽度（默认：128）
- **响应示例**:
```json
{
  "success": true,
  "results": [1, 5, 10, 15, 20],
  "k": 10,
  "ef_search": 128
}
```

## 集群架构

本项目采用存算分离架构：

```
┌─────────────────┐         RDMA          ┌──────────────────┐
│  Compute Node   │◄──────────────────────►│  Memory Server   │
│  (HTTP Server)  │                        │  (Storage)       │
│                 │                        │                  │
│  - HTTP API     │                        │  - Vector Index  │
│  - HNSW Search  │                        │  - RDMA Memory   │
│  - Cache        │                        │                  │
└─────────────────┘                        └──────────────────┘
        ▲
        │ HTTP
        │
┌───────┴─────────┐
│   Client        │
│  (test_api.py)  │
└─────────────────┘
```

## 故障排查

### 问题 1: 无法连接到服务器

**症状**: `Connection refused` 或 `服务器未运行`

**解决方案**:
1. 检查服务器是否启动：`ps aux | grep shine`
2. 检查端口是否被占用：`netstat -tlnp | grep 8080`
3. 检查防火墙设置

### 问题 2: Huge Pages 分配失败

**症状**: `Allocating huge-pages failed`

**解决方案**:
```bash
# 检查当前配置
cat /proc/sys/vm/nr_hugepages

# 设置 Huge Pages（需要 root 权限）
sudo echo 32768 > /proc/sys/vm/nr_hugepages

# 检查可用性
grep -i huge /proc/meminfo
```

### 问题 3: RDMA 设备未找到

**症状**: `No IB devices found`

**解决方案**:
1. 检查 RDMA 设备：`ibv_devices`
2. 安装 RDMA 驱动和库：`sudo apt-get install libibverbs1 libibverbs-dev`
3. 确保硬件支持 RDMA

## 依赖

Python 测试脚本需要安装 `requests` 库：

```bash
pip install requests
```

或者使用项目的 requirements.txt：

```bash
pip install -r ../requirements.txt
```

## 性能测试

运行性能测试以评估系统性能：

```bash
python3 test_api.py --performance --num-vectors 1000
```

性能测试会统计：
- 批量插入耗时
- 平均插入延迟
- 批量查询耗时
- 平均查询延迟

## 更多信息

- 项目主 README: `/home/xjs/experiment/shine-hnsw-index/README.md`
- 配置说明: `/home/xjs/experiment/shine-hnsw-index/scripts/config.py`
- 集群配置: `/home/xjs/experiment/shine-hnsw-index/rdma-library/library/utils.cc`