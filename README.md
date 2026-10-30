
# 从 AI Coding 到 AI 开发团队：Graph的恢复

在上一篇文章中，我们通过引入checkpointer以及Graph的__interrupt__机制，使Graph可以实现中断，完成的审批和继续执行

这不由让我想让我们的工作流支持宕机重启的能力。同时也想了解下LangGraph时如何存储和恢复Graph的运行的

## 一、代码改造


### 引入SqliteSaver保存checkpoint状态

在LangGraph里提供了以下几宗checkpoint saver的实现
MemorySaver
SqliteSaver
AsyncSqliteSaver
PostgresSaver
AsyncPostgresSave

上一次我们使用了MemorySaver，由于 `MemorySaver` 的状态保存在当前进程的内存中。这意味着，只要进程退出或重启，例如服务宕机，云上的Pod漂移，之前保存的状态就会丢失。

这一版我们选用了 `SqliteSaver`，状态借助Sqlite写入磁盘上的一个数据库文件：

```python
@contextmanager
def persistent_checkpointer(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    db_path = checkpoint_db_path(path)
    conn = sqlite3.connect(
        db_path, check_same_thread=False, timeout=BUSY_TIMEOUT_SECONDS
    )
    try:
        saver = SqliteSaver(
            conn,
            serde=JsonPlusSerializer(allowed_msgpack_modules=msgpack_allowlist()),
        )
        saver.setup()
        yield saver
    finally:
        conn.close()
```
可以看到，除了管理数据库连接，其他的代码基本一致。

具体数据库文件路径设置在aidevteam.properties中，别忘了在docker-compose.yml中将Volume挂载出来，否则coontainer重启状态还是会丢失

### 增加list threads入口

执行是通过thread_id恢复的，给我们的AIDevTeam增加list command，用来列出当前所有执行的thread_id和状态

1. 读取数据库，获得当前所有的执行的thread_id
2. 通过LangGraph的Graph的get_state方法重建存储的Graph状态
3. 便利所有的Graph状态，并展示

### 增加Resume入口
1. 通过LangGraph的Graph的get_state方法重建指定thread_id的Graph状态
2. 如果是等待用户输入则进入_approval_loop 循环
3. 如果是恢复其他节点则，通过graph.invoke 继续Graph的执行

这里为了简化即使是其他节点也通过外面嵌套一层_approval_loop

因为恢复的节点有可能是Plan节点，如果不加这一层在Plan执行完之后，会进入interrupt状态，程序会直接结束
···
_approval_loop(graph, config, graph.invoke(None, config))
···

这次改造就完成了，比较简单的一次改动

实际上今天的重头戏是看一下，LangGraph的Graph执行信息是如何存储的

我在等待计划审批时中断了程序执行，让我们看一下数据库里都有些什么



数据库里包含checkpoints和writes两张表，其中已经对checkpoints表的checkpoint字段和writes表的value进行了反序列化

### 表：`checkpoints`
| thread_id | checkpoint_ns | checkpoint_id | parent_checkpoint_id | type | checkpoint | metadata |
|---|---|---|---|---|---|---|
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `NULL` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:50:02.312955+00:00',` | `{'source': 'input', 'step': -1, 'parents': {}}` |
| `20261028-215002-2569` |  | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:50:02.313629+00:00',` | `{'source': 'loop', 'step': 0, 'parents': {}}` |
| `20261028-215002-2569` |  | `1f1d319c-ee88-6ade-8001-28d70e8f396a` | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:52:02.418531+00:00',` | `{'source': 'loop', 'step': 1, 'parents': {}}` |


### 表：`writes`

| thread_id | checkpoint_ns | checkpoint_id | task_id | task_path | idx | channel | type | value |
|---|---|---|---|---|---:|---|---|---|
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `fda54edf-3a0d-0078-9166-043b41b61cbc` | `~__pregel_pull, __start__` | 0 | `task` | `msgpack` | `build a word-count CLI tool` |
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `fda54edf-3a0d-0078-9166-043b41b61cbc` | `~__pregel_pull, __start__` | 1 | `run_id` | `msgpack` | `20261028-215002-2569` |
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `fda54edf-3a0d-0078-9166-043b41b61cbc` | `~__pregel_pull, __start__` | 2 | `iteration` | `msgpack` | `0` |
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `fda54edf-3a0d-0078-9166-043b41b61cbc` | `~__pregel_pull, __start__` | 3 | `plan_approval` | `msgpack` | `True` |
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `fda54edf-3a0d-0078-9166-043b41b61cbc` | `~__pregel_pull, __start__` | 4 | `branch:to:planner` | `null` |  |
| `20261028-215002-2569` |  | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `30216ee0-e441-5363-314d-10d01cc85291` | `~__pregel_pull, planner` | 0 | `plan` | `msgpack` | `app_name='word-count' summary='A small standard-li` |
| `20261028-215002-2569` |  | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `30216ee0-e441-5363-314d-10d01cc85291` | `~__pregel_pull, planner` | 1 | `branch:to:approve_plan` | `null` |  |
| `20261028-215002-2569` |  | `1f1d319c-ee88-6ade-8001-28d70e8f396a` | `8e0622f1-964c-b699-3aff-808c5686e312` | `~__pregel_pull, approve_plan` | -3 | `__interrupt__` | `msgpack` | `[Interrupt(value={'plan': {'app_name': 'word-count'` |


thread_id就是我们的一次执行，checkpoint_id可以对应到一个Graph的节点，checkpoint是我们最感兴趣的
让我们展开最后一个checkpoint
···
    'v': 4, 
    'ts': '2026-10-28T21:52:02.418531+00:00', 
    'id': '1f1d319c-ee88-6ade-8001-28d70e8f396a', 
    'channel_values': {
        'task': 'build a word-count CLI tool', 
        'run_id': '20261028-215002-2569', 
        'plan': Plan(app_name='word-count', summary='A small standard-library-only CLI that counts words in text supplied as arguments or read from files, with optional filtering of short words and support for common plain-text encodings.', cli_commands=['python app.py count "Hello, world!"', 'python app.py count --min-length 3 "a be cat"', 'python app.py count --ignore-case "Hello hello"', 'python app.py count notes.txt', 'python app.py count notes.txt another.txt', 'python app.py count --stdin', 'python app.py count --min-length 3 --ignore-case notes.txt'], files=[FileSpec(path='app.py', description='CLI entry point and reusable functions. Implement count_words(text, min_length=1, ignore_case=False) to split text on whitespace and count words meeting the minimum length; read_text(path) to read UTF-8 text files with a clear error for missing/unreadable files; and build_parser()/main(argv=None) using argparse with a count subcommand. Accept one or more text arguments and/or file paths, plus --stdin to read standard input, --min-length (non-negative integer), and --ignore-case. Print a total word count and, when multiple inputs are supplied, a per-input breakdown. Handle empty input and file errors with friendly messages and non-zero exit codes.')]), 
        'iteration': 0, 
        'plan_approval': True, 
        'branch:to:approve_plan': None
    }, 
    'channel_versions': {
        '__start__': '00000000000000000000000000000002.0.73221403814305', 
        'task': '00000000000000000000000000000002.0.73221403814305', 
        'run_id': '00000000000000000000000000000002.0.73221403814305', 
        'iteration': '00000000000000000000000000000002.0.73221403814305', 
        'plan_approval': '00000000000000000000000000000002.0.73221403814305', 
        'branch:to:planner': '00000000000000000000000000000003.0.45862768924600783', 
        'plan': '00000000000000000000000000000003.0.45862768924600783', 
        'branch:to:approve_plan': '00000000000000000000000000000003.0.45862768924600783'
    }, 
    'versions_seen': {
        '__input__': {}, 
        '__start__': {'__start__': '00000000000000000000000000000001.0.7110444538049118'}, 
        'planner': {'branch:to:planner': '00000000000000000000000000000002.0.73221403814305'}
    }, 
    'updated_channels': ['branch:to:approve_plan', 'plan']
···


- `channel_values`：当前所有状态 channel 的值
- `updated_channels`：这个 checkpoint 中哪些 channel 发生了变化
- `channel_versions`：各 channel 的版本号
- `versions_seen`：各节点已经消费过哪些版本

channel可以理解为当前节点的是个属性？？，
checkpoints表的每一条记录就是一个当前节点的快照，在恢复Graph时，LangGraph可以快速的查找到当前节点，依赖 `channel_values` 中的信息快速重建Graph状态。



而writes表中保存有和他关联的checkpoint的具体对channel值的更改信息，每个channel的一个更改一条记录，依靠writes表可以一步一步的重放Graph的历史

在仔细研究存储内容之前，我以为里面只保留了最后一步的State值，一看才知道是我浅薄了，原来checkpoints的设计不止可以resume中断的状态，还可以一步一步回溯Graph的执行

可以优化改进流程，审计操作，很多的应用场景



# 从 AI Coding 到 AI 开发团队：Graph 的恢复

在上一篇文章中，我们通过引入 Checkpointer 和 Graph 的 `interrupt()` 机制，让 AIDevTeam 可以暂停工作流，等待人工审批，再继续执行。

不过，这还留下了一个问题。

如果工作流暂停后，服务进程意外退出了呢？如果运行任务的 Pod 被重建了呢？如果我想在第二天回来之后，再继续昨天没有完成的任务呢？

这让我想进一步了解 LangGraph 是如何保存和恢复工作流的，也决定给 AIDevTeam 增加宕机重启后的恢复能力。

## 一、代码改造

### 1. 引入 SqliteSaver 保存 checkpoint

LangGraph 提供了多种 Checkpointer 实现，适用于不同的运行环境和持久化需求，例如：

- `InMemorySaver`（也常被称为 `MemorySaver`）：将状态保存在内存中。
- `SqliteSaver`：使用 SQLite 数据库保存状态。
- `AsyncSqliteSaver`：SQLite 的异步版本。
- `PostgresSaver`：使用 PostgreSQL 保存状态。
- `AsyncPostgresSaver`：PostgreSQL 的异步版本。

上一版我们使用的是 `MemorySaver`。由于状态保存在当前进程的内存中，只要进程退出或重启，之前保存的状态就会丢失。

因此，这一版我选择了 `SqliteSaver`，将 checkpoint 写入磁盘上的 SQLite 数据库文件。

核心代码如下：

```python
@contextmanager
def persistent_checkpointer(
    path: str | Path | None = None,
) -> Iterator[SqliteSaver]:
    db_path = checkpoint_db_path(path)

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        timeout=BUSY_TIMEOUT_SECONDS,
    )

    try:
        saver = SqliteSaver(
            conn,
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=msgpack_allowlist()
            ),
        )
        saver.setup()
        yield saver
    finally:
        conn.close()
```

可以看到，除了管理数据库连接和设置序列化器，其他逻辑与之前基本一致。

数据库文件路径配置在 `aidevteam.properties` 中。这里还有一个容易忽略的细节：**数据库文件必须存放在容器重启后仍然存在的位置。**

因此，我还在 `docker-compose.yml` 中配置了 Volume，将数据库文件挂载到持久化存储上。否则，即使使用了 `SqliteSaver`，容器被删除或重新创建后，数据库文件仍然可能丢失。

这样，工作流状态就不再依赖某个进程的内存了。

### 2. 增加 list 命令

既然状态已经持久化，接下来就需要一个入口，让我们知道目前有哪些任务，以及它们处于什么状态。

因此，我给 AIDevTeam 增加了 `list` 命令，主要完成三件事：

1. 从 checkpoint 存储中获取当前存在的工作流线程。
2. 根据对应的 `thread_id`，调用 LangGraph 的 `get_state()` 获取工作流状态。
3. 根据状态快照及其执行信息，整理并展示任务状态。

这里需要区分两个概念：`thread_id` 用于标识一条工作流线程，而 checkpoint 是这条线程在某个时刻保存的状态快照。

通过 `get_state()`，我们可以读取对应线程最近保存的状态，并进一步判断工作流是否已经结束、是否还需要恢复，以及是否存在等待处理的中断。

这让原本只能通过日志观察的任务，开始具备了基本的查询和管理能力。

### 3. 增加 Resume 命令

有了任务列表，下一步就是恢复任务。

我增加了 `resume` 入口，主要流程如下：

1. 根据指定的 `thread_id`，通过 `graph.get_state()` 获取当前工作流状态。
2. 如果工作流正在等待人工输入，就进入 `_approval_loop`，继续处理审批和恢复。
3. 如果工作流处于其他可恢复状态，则调用 `graph.invoke()` 继续执行，并在必要时进入同一个审批循环。

为什么要把执行过程统一放到 `_approval_loop` 中？

因为恢复一个工作流，不代表它一定会直接执行到结束。

例如，某个任务恢复后，可能从 Planner 节点继续执行。Planner 完成后，工作流又会进入 `approve_plan` 节点，并触发下一次 `interrupt()`。

如果外层程序只调用一次 `graph.invoke()`，然后就直接退出，那么新的中断就没有机会继续交给用户处理。

所以，我把执行和处理中断的逻辑统一封装在 `_approval_loop` 中：

```python
_approval_loop(
    graph,
    config,
    graph.invoke(None, config),
)
```

这里的 `graph.invoke(None, config)` 用于从已有的工作流状态继续执行；`_approval_loop` 则负责处理执行结果中可能出现的中断。

具体实现仍然需要区分首次启动、等待审批、普通恢复和任务结束等情况，但对外可以统一成一个恢复入口。

到这里，代码改造基本完成了。

从实现上看，这只是把内存存储换成了 SQLite，再增加两个管理命令，并没有特别复杂的改动。

不过，真正有意思的部分才刚刚开始。

## 二、看看数据库里究竟保存了什么

我原本以为，LangGraph 的 checkpoint 只需要保存最后一次的 State 就够了。

为了验证这个想法，我启动了一个任务，在等待计划审批时中断了程序，然后直接查看 SQLite 数据库里的内容。

结果发现，LangGraph 保存的信息比我预想的要丰富得多。

### 1. checkpoints 表

在当前使用的 SQLite Checkpointer 实现中，主要可以看到 `checkpoints` 和 `writes` 两张表。

先看 `checkpoints` 表。下面是一次任务中几条记录的简化示例，`checkpoint` 字段只展示了部分内容。

| thread_id | checkpoint_ns | checkpoint_id | parent_checkpoint_id | type | checkpoint | metadata |
|---|---|---|---|---|---|---|
| `20261028-215002-2569` |  | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `NULL` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:50:02.312955+00:00',` | `{'source': 'input', 'step': -1, 'parents': {}}` |
| `20261028-215002-2569` |  | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `1f1d3198-751e-6214-bfff-692d7ccf9fea` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:50:02.313629+00:00',` | `{'source': 'loop', 'step': 0, 'parents': {}}` |
| `20261028-215002-2569` |  | `1f1d319c-ee88-6ade-8001-28d70e8f396a` | `1f1d3198-751f-6c6e-8000-dda7f0ef6d01` | `msgpack` | `{'v': 4, 'ts': '2026-10-28T21:52:02.418531+00:00',` | `{'source': 'loop', 'step': 1, 'parents': {}}` |


其中有几个值得关注的字段：

- `thread_id`：标识这条工作流线程。
- `checkpoint_ns`：标识 checkpoint 所属的命名空间。简单的 Graph 通常为空；使用子图时，可能会出现不同的命名空间。
- `checkpoint_id`：标识一份具体的 checkpoint 快照。
- `parent_checkpoint_id`：记录父 checkpoint 的标识，可以用来关联不同快照。
- `metadata`：记录 checkpoint 的来源、执行步骤等元信息。

这里要特别说明：**`checkpoint_id` 标识的是一份状态快照，而不是一个 Graph 节点。** 一个节点的执行过程可能涉及多份 checkpoint，不能简单地把它们一一对应。

### 2. writes 表

再看 `writes` 表。这里保存的是与某次 checkpoint 关联的任务写入信息。

例如，在我的任务中，可以看到类似下面的记录：


| checkpoint_id | task_path | idx | channel | value（简化） |
|---|---|---:|---|---|
| 初始 checkpoint | `__start__` | 0 | `task` | `build a word-count CLI tool` |
| 初始 checkpoint | `__start__` | 1 | `run_id` | `20261028-215002-2569` |
| 初始 checkpoint | `__start__` | 2 | `iteration` | `0` |
| Planner checkpoint | `planner` | 0 | `plan` | `Plan(...)` |
| Planner checkpoint | `planner` | 1 | `branch:to:approve_plan` | `null` |
| 审批 checkpoint | `approve_plan` | -3 | `__interrupt__` | `[Interrupt(...)]` |

为了方便阅读，我省略了部分字段和序列化细节。实际表中还会记录 `task_id`、`checkpoint_ns`、`type` 等信息。

从这些数据中，我们可以看到工作流的输入、Planner 生成的计划，以及审批节点产生的中断信息。

这也解释了为什么 `checkpoints` 和 `writes` 需要配合使用：前者保存状态快照，后者记录执行过程中与任务相关的写入信息。它们共同参与工作流状态的保存和恢复，而不是简单地分别承担“当前状态”和“完整历史日志”两个互不相关的职责。

## 三、展开最后一个 checkpoint

真正让我感兴趣的是 `checkpoint` 字段。

对它进行反序列化后，可以看到类似下面的结构。为了突出重点，我省略了部分版本号和内部细节：

```python
{
    "v": 4,
    "ts": "2026-10-28T21:52:02.418531+00:00",
    "id": "1f1d319c-ee88-6ade-8001-28d70e8f396a",

    "channel_values": {
        "task": "build a word-count CLI tool",
        "run_id": "20261028-215002-2569",
        "plan": Plan(...),
        "iteration": 0,
        "plan_approval": True,
        "branch:to:approve_plan": None,
    },

    "channel_versions": {
        "task": "...",
        "run_id": "...",
        "plan": "...",
        "iteration": "...",
    },

    "versions_seen": {
        "__start__": {...},
        "planner": {...},
    },

    "updated_channels": [
        "branch:to:approve_plan",
        "plan",
    ],
}
```

这份结构里有几个比较重要的字段。

### 1. channel_values：当前保存的状态值

`channel_values` 保存的是 checkpoint 中各个状态 channel 对应的值。

在我们的 AIDevTeam 中，可以看到任务输入、`run_id`、开发计划、迭代次数等信息。

这里可以把 channel 理解为 LangGraph 状态中的一个数据通道。它不完全等同于普通对象的属性：除了用户定义的 State 字段，LangGraph 内部还有用于调度和执行控制的 channel。

因此，`channel_values` 不只是我们在 `TeamState` 中声明的业务字段，还可能包含工作流执行所需的内部信息。

### 2. updated_channels：这次更新了什么

`updated_channels` 记录这份 checkpoint 中发生更新的 channel。

例如，这次保存的状态中包含 `plan` 和 `branch:to:approve_plan`，说明这一步的执行涉及计划数据和工作流分支信息。

通过这个字段，可以更方便地了解某份 checkpoint 与前后执行步骤之间的关系。

### 3. channel_versions：各个 channel 的版本

`channel_versions` 记录各个 channel 的版本信息。

LangGraph 使用这些版本信息来追踪状态变化，并判断不同节点和 channel 之间的依赖关系。

它们不是简单的业务版本号，也不应该被直接理解为某个字段的修改次数。

### 4. versions_seen：节点已经处理过哪些版本

`versions_seen` 记录各个节点已经处理过的 channel 版本。

这与 LangGraph 的执行调度有关：框架需要知道节点已经消费过哪些状态版本，才能判断后续是否需要再次执行。

这也是为什么 checkpoint 里不只有业务数据，还包含了不少看起来比较底层的执行元信息。

**LangGraph 保存的不是单纯的一份 State，而是恢复工作流所需要的状态和调度信息。**

## 四、原来 Graph 不只可以恢复，还可以回溯

研究完这些字段后，我对 LangGraph 的 checkpoint 机制有了更深入的认识。

一开始，我以为它只是在工作流中断时保存最后一份 State，等恢复时再把 State 读出来。

但从数据库结构可以看出，LangGraph 还会保存多份关联的 checkpoint，并通过 `parent_checkpoint_id` 等信息将它们联系起来。

这意味着，我们不仅可以恢复某条工作流线程，还可以利用保存的历史状态，检查工作流在不同时间点的执行情况。

例如，LangGraph 提供的状态查询能力可以用于读取当前状态，也可以在支持的接口中查看历史 checkpoint。基于这些能力，我们可以进一步探索：

- **调试**：检查某一步执行时，工作流中保存了哪些状态。
- **审计**：追踪计划、反馈和其他状态是如何变化的。
- **状态回溯**：查看工作流在较早的 checkpoint 中是什么状态。
- **流程优化**：分析工作流在哪些节点反复执行，或者在哪些环节频繁失败。

当然，这不意味着 `writes` 表本身就是一份可以任意重放的完整事件日志，也不意味着只要恢复到旧 checkpoint，就能撤销已经发生的外部操作。

例如，AI 如果已经调用外部 API 创建了资源，或者已经修改了真实系统中的配置，恢复到旧 checkpoint 并不会自动撤销这些操作。

**状态可以回溯，不代表外部世界也能回滚。**

这也是为什么在 AIDevTeam 中，我们既需要可靠的状态持久化，也需要继续做好执行权限控制、操作审计和必要的人工审批。

下一步，我想继续探索的是：在已经具备持久化和恢复能力的基础上，能否进一步利用这些机制，让工作流更容易调试、更容易观察，也更容易从失败中恢复。

毕竟，让 AI 自动完成任务只是第一步。

让它能够在服务重启后继续工作，并且让我们看清它究竟做过什么，才更接近一个真正可用的 AI 开发团队。