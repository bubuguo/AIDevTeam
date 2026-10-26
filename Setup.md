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

流水线通过容器运行（见下节「容器化运行」）。`python -m aidevteam.cli graph` 可在本地
打印流水线 Mermaid 图（构造图只是包装远程节点引用，不需要节点服务在线）。

---

## 容器化运行（podman / docker）

每个 Agent 节点运行在独立容器中，LangGraph 编排器（orchestrator）通过 HTTP 异步任务
模式（提交 + 轮询，见 `src/aidevteam/rpc.py`）驱动它们。LLM 生成的代码只在 tester
容器内执行，与宿主机隔离。`Dockerfile` / `docker-compose.yml` 同时兼容 podman 与 docker。

```bash
# 前置：项目根目录需要有 .env（LLM 端点配置），并创建 generated/ 目录
mkdir -p generated

# 构建镜像并启动 4 个节点服务（podman 与 docker 命令相同，二选一）
podman compose up -d --build planner coder tester reviewer
# docker compose up -d --build planner coder tester reviewer

# 按需运行流水线（orchestrator 不是常驻服务）
podman-compose run --rm orchestrator run "写一个把摄氏度转华氏度的命令行工具"
# docker 用户：docker compose run --rm orchestrator run "..."

# 只打印流水线 Mermaid 图（不需要 LLM）
podman-compose run --rm orchestrator graph

# 观察某个节点的日志 / 停止全部服务
podman compose logs -f tester
podman compose down
```

说明：

- **podman 用户注意**：`podman compose run`（包装器）在 macOS 上有 EINTR bug，`run`
  子命令请直接使用 `podman-compose run`；`up`/`logs`/`down` 等不受影响。
- **workspace 共享**：`./generated` 以 bind mount 挂载为容器内 `/workspace`，coder 写入、
  tester 执行测试，产物直接落在宿主机 `./generated/<时间戳>/`。
- **新增节点类型**：在 `agents/<name>.py` 定义 `<name>(state)` 并在 `build_graph` 里
  `add_node` 接线，再加一段 compose 服务即可——服务 URL 由角色名自动推导
  （`http://<role>:8000`），无需任何 URL 配置。
