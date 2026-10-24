## 环境搭建

### 前置要求

- Python ≥ 3.11

### 方式一：uv（推荐）

```bash
# 安装 uv
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 一键创建 .venv 并安装全部依赖（含 dev 组的 pytest）
uv sync

# 运行（uv 自动使用项目 venv，无需手动激活）
uv run aidevteam --help
uv run python -m aidevteam.cli --help
```

uv 会根据 `uv.lock` 精确还原依赖版本，且始终指向当前项目的 venv，不依赖 shell 的激活状态。

### 方式二：pip + venv

```bash
# 在项目根目录下创建并激活虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 安装项目及运行时依赖（editable 模式，同时注册 aidevteam 命令）
pip install -e .

# 可选：安装 pytest 等开发依赖（需要 pip ≥ 25.1，PEP 735 dependency-groups）
pip install -e . --group dev

# 运行（以下两种写法等价）
aidevteam --help
python -m aidevteam.cli --help
```

### 配置 LLM 端点

在项目根目录创建 `.env` 文件（任一 OpenAI 兼容端点均可）：

```bash
LLM_BASE_URL=https://api.example.com/v1   # 必填
LLM_MODEL=model-id                   # 必填
LLM_API_KEY=sk-...                        # 可选，本地 Ollama 等无需鉴权时可省略
```

缺失必填项时程序会在启动时立即报错并提示。运行时调优可写入可选的 `aidevteam.properties`（Java 风格 key=value）：

```properties
temperature=0.2
max_iterations=3
```

配置优先级（高 → 低）：CLI flags > 环境变量 / `.env` > `aidevteam.properties` > 默认值。

### 运行示例

```bash
# 打印 Agent 流水线的 Mermaid 图
python -m aidevteam.cli graph

# 驱动 AI 开发团队构建一个 Python CLI 应用
python -m aidevteam.cli run "写一个 CSV 转 JSON 的命令行工具"
```
