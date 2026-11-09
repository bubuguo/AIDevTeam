
# 从 AI Coding 到 AI 开发团队：接入外部需求管理

在上一篇文章中，我们通过对使用数据库代替MemorySaver使得我们的AI团队可以从宕机，重启等异常条件中继续进行工作。鲁棒性有了一大步的提升
为进行大规模开发奠定了第一步基础

但是且慢，为了使我们的AI团队更加正规，这周将需求管理列为改造的重点。自然Github的提供的Issue功能因为其超高的性价比（免费）成了我的选择

## 一、代码改造

### 1. 定义协议

本次改造的核心是建立统一的协议，明确内部工作流与外部GitHub Issue的交互方式。经过深入分析，我们放弃了最初使用Agent+MCP调用GitHub API的想法，转而采用确定的REST API行为，原因如下：

1. **不要为用Agent+MCP而用Agent+MCP**：模型驱动的函数调用往往缺乏确定性，同一请求可能得到不同的结果，难以保证行为的可预测性。

2. **确定的行为使用确定的API**：将每一个具体的操作（创建、查询、更新、评论）都映射到具体的HTTP方法和端点，避免模型对API的误解或错误调用。

3. **安全无歧义**：通过API的认证和权限控制，而不是依赖模型对API参数的理解，消除因模型解释不一致而产生的安全风险。

4. **可追踪性**：所有的需求操作都有明确的请求路径和响应结构，便于日志记录、审计和调试。

**定义的协议体系**：

**Issue 标签体系**：
- `requirement`：功能需求标签，区分不同类型的开发需求
- `task`：具体任务标签，用于跟踪任务的执行状态
- `approval`：人工审批标签，标记需要人工介入的节点

**Issue 字段定义**：
```yaml
title: "[需求] <简短描述>"
body:
  - label: "需求类型"
    description: "需求的大类（如: feature, bug, improvement）"
  - label: "优先级"
    description: "P0/P1/P2/P3，表示优先级别"
  - label: "预期产出"
    description: "期望生成的功能或代码的简要描述"
  - label: "接受标准"
    description: "验收任务的具体标准，用于判断任务是否完成"
```

**REST API 行为约定**：
- `GET /repos/{owner}/{repo}/issues`：列出项目下的需求Issue
- `POST /repos/{owner}/{repo}/issues`：创建新的需求Issue
- `GET /repos/{owner}/{repo}/issues/{issue_number}`：获取指定Issue的详细信息
- `PATCH /repos/{owner}/{repo}/issues/{issue_number}`：更新Issue的标签、状态或其他属性
- `POST /repos/{owner}/{repo}/issues/{issue_number}/comments`：在Issue中添加评论，用于节点间的反馈交流

### 2. 优化State

这次改造的核心副产品是：**将需求信息从内部State中剥离，存入GitHub Issue中作为中期记忆。**

**优化前的问题**（v1-v3版本）：
- TeamState承载了任务的全部上下文：从用户需求、开发计划到迭代反馈
- 每次工作流重启或线程切换时，都需要重新搬运这些信息
- 消耗宝贵的LLM token，且上下文容易被各类执行细节稀释
- 随任务复杂度增加，信息检索和重用变得日益困难

**优化后的设计**：

**State 中不再存储的内容**：
- 完整的需求文本描述
- 冗余的执行历史和中间结果
- 与Issue关联的元数据（这些现在由GitHub Issue管理）

**State 中保留的精简内容**：
- **Issue 引用**：Issue的URL或编号，作为指向外部记录的指针，占用极少token
- **上下文摘要**：需求的关键要点摘要（1-2句话），用于快速定位和判断是否需要拉取完整信息
- **执行状态**：当前任务在工作流中的位置（plan、code、test、review等），用于协调工作流流转
- **迭代计数**：当前的迭代次数，用于跟踪修改历史和避免无限循环
- **临时执行上下文**：当前节点正在处理的短期信息，如当前测试的具体错误模式

**State 结构优化示例**：

```python
class TeamState(TypedDict, total=False):
    """精简后的工作流状态，仅包含执行必需的核心信息"""
    
    # 指向外部记录的引用
    issue_number: int  # GitHub Issue编号
    issue_url: str     # GitHub Issue的URL
    
    # 核心执行状态
    task: str          # 简短的任务描述（不超过50字符）
    iteration: int     # 当前迭代次数
    approved: bool     # 是否通过审查
    
    # 执行过程中的临时信息
    current_focus: str # 当前节点关注的点
    last_feedback: str # 上一次的反馈摘要
```

**优化带来的效果**：
- 单次上下文长度控制在合理范围（例如 < 8k tokens）
- 需求信息不再因工作流重启而丢失或稀释
- 新成员可以通过Issue历史快速了解项目背景
- 真正实现“context时最贵的资源”这一原则的落地

### 3. CLI 改造

为了支持Issue驱动的工作流，CLI需要新增几个核心命令，并重构原有的工作流入口。

**新增命令结构**：

```bash
# 创建新需求
AIDevTeam$ python -m aidevteam.cli requirement create "创建 JSON 格式化工具"
# 系统后台自动：
# 1. 调用 REST API: POST /repos/{owner}/{repo}/issues
# 2. 返回Issue编号和URL
# 3. 标记相应的label（requirement, task等）
# 4. 输出创建结果给用户

# 列出当前活跃需求
AIDevTeam$ python -m aidevteam.cli requirement list
# 1. 读取数据库，获得当前所有执行的thread_id
# 2. 通过Graph的get_state方法重建存储的Graph状态
# 3. 便利所有的Graph状态，并展示Issue编号、当前状态、迭代次数
# 4. 输出格式化的需求列表

# 根据 Issue 启动工作流
AIDevTeam$ python -m aidevteam.cli run --issue 123
# 1. 通过 REST API GET /repos/{owner}/{repo}/issues/123 获取Issue内容
# 2. 解析关键信息（需求描述、优先级、接受标准等）
# 3. 将解析结果同步到工作流 State 中：
#    - state["issue_number"] = 123
#    - state["issue_url"] = "https://github.com/.../issues/123"
#    - state["task"] = parsed_requirement_summary
# 4. 启动工作流：Plan → Code → Test → Review
# 5. 每个节点都可以通过Issue评论进行反馈和协作

# 更新需求状态
AIDevTeam$ python -m aidevteam.cli requirement update 123 --status "approved"
# 1. CLI调用 REST API: PATCH /repos/{owner}/{repo}/issues/123
# 2. 请求体: {"state": "closed", "labels": ["completed"]}
# 3. Issue关闭，任务标记为 completed
# 4. 记录本次更新的时间和原因

# 查看 Issue 详情
AIDevTeam$ python -m aidevteam.cli requirement show 123
# 1. 获取Issue详细信息
# 2. 显示当前工作流状态
# 3. 展示相关的评论历史
# 4. 显示迭代进度

# 恢复中断的工作流
AIDevTeam$ python -m aidevteam.cli run --issue 123 --resume 3
# 1. 自动检测Issue #123是否有关联的工作流状态
# 2. 如果Issue仍然打开：询问是否继续进行中的任务
# 3. 如果Issue已关闭：询问是否创建新任务或重新打开
# 4. 恢复到第3次迭代的状态，同步当前 State: iteration=3
```

**CLI 与工作流的集成**：

CLI 不再是单纯的任务启动入口，而是成为了管理外部需求与内部工作流之间桥梁的工具：

1. **需求同步**：CLI 负责将GitHub Issue的内容解析并同步到当前的工作流 State中，确保工作流始终基于最新的需求信息。

2. **状态持久化**：通过Issue的状态（打开/关闭、标签）作为工作流持久化的入口，即使工作流进程重启、Pod漂移，Issue记录不受影响。

3. **反馈闭环**：CLI 协调各个节点（Planner、Coder、Tester、Reviewer）通过Issue评论进行反馈，形成完整的闭环。

4. **跨会话延续**：用户可以在不同的会话中通过Issue Key快速恢复上下文，无需依赖内存状态。

**CLI 工作流示例**：

```mermaid
flowchart TD
    A[CLI 命令] --> B{解析命令}
    B -->|requirement create| C[调用 REST API 创建 Issue]
    B -->|requirement list| D[读取数据库获取 thread_id]
    B -->|run --issue N| E[同步 Issue 内容到 State]
    B -->|requirement update N --status S| F[PATCH API 更新 Issue]
    
    C --> G[返回 Issue 编号和 URL]
    D --> H[列出活跃需求列表]
    E --> I[同步关键信息到 State]
    I --> J[启动工作流：Plan → Code → Test → Review]
    J --> K[各节点通过 Issue 评论反馈]
    K --> L[工作流完成或需要迭代]
    L -->|迭代| I
    L -->|完成| M[关闭 Issue]
    
    style A fill:#e1f5fe
    style M fill:#c8e6c9
</flowchart>

## 二、需求管理演进

随着 AI 开发团队的迭代，我们从最基础的代码生成工作流，逐步意识到**上下文管理**是支撑大规模开发的关键瓶颈。之前的版本中，我们通过容器化和状态持久化解决了代码执行的安全和恢复问题，但所有的需求信息仍然散落在工作流的 State 中，随着任务复杂度增加，信息检索和重用变得日益困难。

### 1. 从 State 中抽离需求记录

在 v1-v3 的版本中，我们的 TeamState 承载了任务的全部上下文：从用户需求、开发计划到迭代反馈。然而，这种设计意味着每次工作流重启或线程切换时，都需要重新搬运这些信息，不仅消耗宝贵的 token，也使得上下文容易被各类执行细节稀释。

这次改造的核心副产品是：**将需求信息从 State 中剥离，存入 GitHub Issue 中作为中期记忆。**

这样带来的直接好处是：
- **Context 成本下降**：不再需要将完整的需求描述持续累加到每一次 LLM 调用的上下文中
- **信息不被稀释**：GitHub Issue 作为独立的记忆载体，保持需求描述的纯凹和完整性
- **跨会话延续**：即使工作流进程重启，需求记录也能通过 Issue Key 快速恢复，无需依赖内存状态

### 2. 协议定义

为了确保 GitHub Issue 与内部工作流能够无缝交互，我们定义了统一的协议：

**Issue 标签体系**：
- `requirement`：功能需求标签
- `task`：具体任务标签
- `approval`：人工审批标签

**Issue 字段定义**：
```yaml
title: "[需求] <简短描述>"
body:
  - label: "需求类型"
    description: "需求的大类"
  - label: "优先级"
    description: "P0/P1/P2/P3"
  - label: "预期产出"
    description: "期望生成的功能或代码"
  - label: "接受标准"
    description: "验收的标准"
```

**REST API 行为约定**：
- `GET /repos/{owner}/{repo}/issues`：列出相关需求
- `POST /repos/{owner}/{repo}/issues`：创建新需求
- `GET /repos/{owner}/{repo}/issues/{issue_number}`：获取需求详情
- `PATCH /repos/{owner}/{repo}/issues/{issue_number}`：更新需求状态

### 3. CLI 改造

为了支持 Issue 驱动的工作流，CLI 需要新增几个核心命令：

```bash
# 创建新需求
AIDevTeam$ python -m aidevteam.cli requirement create "创建 JSON 格式化工具"

# 列出当前活跃需求
AIDevTeam$ python -m aidevteam.cli requirement list

# 根据 Issue 启动工作流
AIDevTeam$ python -m aidevteam.cli run --issue 123

# 更新需求状态
AIDevTeam$ python -m aidevteam.cli requirement update 123 --status "approved"
```

## 三、技术实现

### 1. 协议定义

我们不再盲目使用 Agent+MCP 来调用 GitHub API，而是采用明确的 REST API 行为：

- **确定的行为使用确定的 API**：每一个具体的操作（创建、查询、更新、评论）都映射到具体的 HTTP 方法和端点
- **安全无歧义**：通过 API 的认证和权限控制，而不是依赖模型的函数调用能力
- **可追踪性**：所有的需求操作都有明确的请求路径和响应结构，便于日志记录和审计

### 2. State 优化

在 State 中我们不再存储完整的需求文本，而是存储：

- **Issue 引用**：Issue 的 URL 或编号，作为指向外部记录的指针
- **上下文摘要**：需求的关键要点摘要，用于快速定位
- **执行状态**：当前任务在工作流中的位置（plan、code、test、review 等）
- **迭代计数**：当前的迭代次数，用于跟踪修改历史

这样设计使得 State 不再膨胀，而成为一个轻量级的执行上下文，真正实现“context 时最贵的资源”这一原则的落地。

### 3. CLI 改造

CLI 的改造重点是将 GitHub Issue 的操作流程化：

**新增命令结构**：
- `requirement:*`：Issue 管理相关命令
- `workflow:*`：工作流控制相关命令
- `status:*`：任务状态查询相关命令

**与工作流的集成**：
CLI 不再是单纯的任务启动入口，而是成为了管理外部需求与内部工作流之间桥梁的工具。用户可以通过 CLI 查看 Issue 状态，或者将 Issue 信息同步到当前的工作流 State 中。

## 四、上下文管理策略

这次改造的另一个重要目标是解决“context 使用过多，冲淡各种信息”问题。

**分层上下文策略**：

1. **核心上下文**（始终包含在 State 中）：
   - 当前任务的执行状态
   - 最近的反馈和迭代信息
   - 正在进行的计划要点

2. **需求上下文**（通过 Issue 引用获取）：
   - 完整的需求描述存储在 GitHub Issue 中
   - 通过 `--issue` 参数或 `requirement get` 命令快速拉取
   - 只在需要时加载，避免永久性累积

3. **历史上下文**（归档记录）：
   - 完成任务的 Issue 关闭后，保持记录可查
   - 通过 Issue 搜索功能回溯历史决策

**效果**：
- 单次上下文长度控制在合理范围（例如 < 8k tokens）
- 需求信息不再因工作流重启而丢失或稀释
- 新成员可以通过 Issue 历史快速了解项目背景

## 五、实际使用流程

### 1. 创建需求 Issue

```bash
AIDevTeam$ python -m aidevteam.cli requirement create "实现一个基于规则的 JSON 格式化工具"
# 系统自动创建 Issue 并标记相应标签
# CLI 输出 Issue URL，例如：https://github.com/.../issues/42
# 在后台，调用 GitHub REST API: POST /repos/{owner}/{repo}/issues
# 返回: { "number": 42, "html_url": "https://github.com/.../issues/42", "title": "...", "labels": [...] }
```

### 2. 根据 Issue 启动工作流

```bash
AIDevTeam$ python -m aidevteam.cli run --issue 42
# CLI 执行以下步骤：
# 1. 通过 REST API GET /repos/{owner}/{repo}/issues/42 获取 Issue 内容
# 2. 解析关键信息（需求描述、优先级、接受标准等）
# 3. 将解析结果同步到工作流 State 中：
#    - state["task"] = parsed_requirement_summary
#    - state["issue_number"] = 42
#    - state["issue_url"] = "https://github.com/.../issues/42"
# 4. 启动工作流：Plan → Code → Test → Review
# 工�流按照 Plan → Code → Test → Review 的流程执行
# 每个节点都可以通过 Issue 评论进行反馈和协作
# 审查节点会自动在 Issue 中添加评论，记录测试结果和审查意见
```

### 3. 迭代与反馈

**测试/审查发现问题时的自动流程**：

1. **Tester 节点**：
   - 运行生成的代码并执行测试套件
   - 如果测试失败，自动在 Issue 中创建评论
   - 评论格式：`🐛 Test Failed: <错误信息>`，包含回溯信息和失败的测试用例
   - 例如：`🐛 Test Failed: json.tool encountered error on line 5: Expecting value: line 1 column 1 (got: '<')`

2. **Reviewer 节点**：
   - 分析代码质量和测试结果
   - 如果发现问题，在 Issue 中添加审查评论
   - 评论格式：`👁️ Review: <具体反馈>`，可能包括代码风格问题、安全隐患、逻辑建议
   - 可以使用 GitHub 的审查功能，发出 `changes_requested` 或 `approved` 状态

3. **Coder 收到反馈**：
   - Coder 节点定期检查 Issue 中的新评论
   - 根据评论内容解析出具体的修改需求
   - 更新计划：`plan["feedback"] = "<从Issue评论中提取的反馈>"`
   - 重新进入 Code 节点，根据反馈修改代码
   - 修改后的代码再次提交，循环进入 Test → Review 环节

**完成后的流程**：

- 如果所有测试通过且审查通过：
  - Coder 更新 Issue 状态：`python -m aidevteam.cli requirement update 42 --status "approved"`
  - CLI 调用 REST API: PATCH /repos/{owner}/{repo}/issues/42
  - 请求体: `{ "state": "closed", "labels": ["completed"] }`
  - Issue 关闭，任务标记为 completed

- 如果需要继续迭代：
  - 保持 Issue 开放状态
  - 下一次 `run --issue 42` 将继续从上次中断的位置继续
  - 工作流状态通过 Issue 的 comment 历史完整记录

### 4. 工作流持久化

**跨进程/跨重启的状态恢复机制**：

1. **Issue 作为持久化入口**：
   - Issue 的状态（打开/关闭、标签）始终存储在 GitHub
   - 即使工作流容器重启、Pod 漂移，Issue 记录不受影响
   - 通过 Issue Key（Issue #42）永久标识该任务

2. **状态恢复流程**：

   ```bash
   # 情况 A：工作流被中断或重启，需要恢复
   AIDevTeam$ python -m aidevteam.cli run --issue 42
   # CLI 自动检测：Issue #42 存在，且有关联的工作流状态
   # - 如果 Issue 仍然打开：询问是否继续进行中的任务
   # - 如果 Issue 已关闭：询问是否创建新任务或重新打开
   
   # 情况 B：明确恢复指定迭代
   AIDevTeam$ python -m aidevteam.cli run --issue 42 --resume 3
   # CLI 恢复到第3次迭代的状态
   # 同步当前 State: iteration=3, 以及相关的 plan/code/test 结果
   ```

3. **Issue comment 作为执行日志**：

   - 所有节点的执行结果、错误、反馈都通过 Issue comment 永久记录
   - 便于事后审计和调试
   - 格式统一：`🔧 <节点名>: <简述>`，例如 `🔧 Coder: 根据反馈修复了 JSON 解析错误`
   - comment 历史可用于：回溯执行路径、分析失败原因、培训新成员

4. **跨线程协作**：

   - 多个并行进行的任务可以使用不同的 Issue Key
   - 同一个任务的不同线程可以通过 Issue 评论协调
   - 例如，一个任务的 Coder 完成后，可以在 Issue 中 @ 另一个任务的 Reviewer 进行代码审查

5. **状态恢复的完整性检查**：

   在恢复工作流前，CLI 会验证：
   - Issue 是否仍然存在且可访问
   - Issue 的当前状态是否允许恢复（是否关闭、是否被标记为 completed）
   - 本地缓存的 State 与 Issue 记录是否一致
   - 如果不一致，优先使用 Issue 记录作为真实来源

### 6. 完整工作流示例

```mermaid
flowchart TD
    A[创建 Issue] --> B[CLI 同步 Issue 内容]
    B --> C{工作流启动}
    C -->|Plan| D[生成开发计划]
    D -->|Code| E[生成代码]
    E -->|Test| F[运行测试]
    F -->|通过| G[审查通过]
    F -->|失败| H[在 Issue 添加评论]
    H -->|Coder 反馈| I[修改代码]
    I -->|重新 Test| F
    G -->|更新 Issue 状态| J[关闭 Issue]
    J --> K[任务完成]
    
    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style K fill:#c8e6c9
</mermaid>

## 六、总结

这次需求管理改造的核心思想是：**将不确定的、易变的外部信息（需求文档、反馈意见）从内部 State 中剥离，转而依靠 GitHub Issue 这种成熟、免费且可追踪的载体。**

**主要收益**：
1. **Context 成本显著降低**：不再需要在每次 LLM 调用中携带完整的需求描述
2. **信息持久性增强**：GitHub Issue 成为开发的中期记忆，跨进程、跨重启均可恢复
3. **行为明确安全**：使用 REST API 替代 Agent+MCP，避免了模型行为的不确定性
4. **扩展性更好**：为后续的按需加载、多线程并行执行奠定了基础

**下一步计划**：
- 完善 REST API 封装层，统一处理认证、重试和错误转换
- 为 Planner 节点设计 Issue 感知的上下文加载逻辑
- 实现 Issue 到 State 的自动映射和反向映射
- 验证跨重启场景下的上下文恢复效果

## 附录：协议一览

| 操作 | 方法 | 端点 | 说明 |
|------|------|------|------|
| 创建需求 | POST | /repos/{owner}/{repo}/issues | 创建新的需求 Issue |
| 列出需求 | GET | /repos/{owner}/{repo}/issues | 列出项目下的 Issue |
| 获取详情 | GET | /repos/{owner}/{repo}/issues/{number} | 获取指定 Issue 的详细信息 |
| 更新状态 | PATCH | /repos/{owner}/{repo}/issues/{number} | 更新 Issue 的标签、状态等 |
| 添加评论 | POST | /repos/{owner}/{repo}/issues/{number}/comments | 在 Issue 中添加评论 |

---

*本次改造承接自 v1-v4 系列文章的经验累积：v1 验证了核心工作流闭环，v2 引入了容器化安全边界，v3 增加了人工审批机制，v4 实现了 checkpoint 持久化。本次改造将重点放在外部需求的管理与内部工作流的协同上，使得 AI 开发团队不仅能够可靠地执行任务，还能够从海量的开发需求中高效地检索和复用上下文。*