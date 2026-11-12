# T03 安装、测试与运行边界

## 通用 CPU 软件测试（Python 3.12）

```sh
python3.12 -m venv /tmp/digital-detective-cpu
/tmp/digital-detective-cpu/bin/python -m pip install -r requirements.lock
/tmp/digital-detective-cpu/bin/python -m pip check
/tmp/digital-detective-cpu/bin/python scripts/test_cpu.py --junitxml=/tmp/dd-cpu-tests.xml
```

`requirements.lock` 是全部固定版本的CPU环境，补齐 numpy、safetensors、outlines_core，并显式包含训练配置读取所需PyYAML。`requirements.txt` 为可读范围声明，复现用lock。CPU环境不安装MLX、MLX-LM或模型权重；测试入口禁止真实MLX导入及主动socket连接，模拟模型仍可用于软件测试。CI在Ubuntu/Python 3.12全新安装并上传JUnit日志。

本机从空环境 `/private/tmp/dd-research-v2-cpu` 安装通过，`pip check`无冲突。软件测试结果与CI链接见任务账本/validation报告，原始日志保留 `artifacts/research_v2/`。Starlette/httpx有弃用提示，尚不影响当前测试；不为消除提示擅自迁移HTTP客户端。

## Apple Silicon 本机运行

首次联网准备按照README，用独立 `.venv` 安装 `requirements-mac.lock` 并下载、校验既定模型。不能用CPU测试环境代替Mac推理环境。已准备机器：

```sh
.venv/bin/python scripts/check_local_setup.py --verify
sh run_offline.sh
# 等价入口
sh run.sh
```

两个入口均强制本地后端和离线环境变量，不自动创建环境、安装依赖或下载模型。缺少准备时直接提示并退出。启动参数固定回环地址127.0.0.1:8765。只读预检本轮通过，确认五案模型；本轮不运行真实模型生成或正式研究测试。针对入口的测试验证缺失依赖时不安装及传给预检/服务器的离线变量。

## Docker

当前Linux Dockerfile只提供CPU Web/软件测试用途；没有Apple Metal，不能作为MLX推理或训练交付。没有本地模型的分析请求应报不可用，不能回退云端。没有在本轮构建Docker镜像。日常Mac使用上述离线入口。
