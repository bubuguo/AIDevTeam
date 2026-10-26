# 从 AI Coding 到 AI 开发团队：容器化，给 AI 加一道安全边界

在上一篇文章中，我们完成了 AIDevTeam 的第一版 Toy。

通过 LangGraph，我们将 Planner、Coder、Tester 和 Reviewer 组织成了一个完整的工作流。现在，只需要从命令行输入开发需求，AI 就可以自动制定计划、生成代码、执行测试，并根据测试和审查结果决定是否需要继续修改。

第一版的基本目标已经达成。

但当 AI 开始自动执行操作时，一个新的问题也逐渐浮现出来：

**如果 AI 做出了错误的判断，甚至执行了超出预期的操作，我们该如何限制它造成的影响？**

## 零、先讲一个真实的故事

我们在 OpenShift 上有一类自定义的 Engine 资源，Controller 会根据其中定义的字段，维持底层资源的配置。

前段时间，我尝试创建了一个 Skill，希望 Hermes 能帮我生成 Patch 结构体，并将 Patch 应用到 Engine 上。

刚开始，一切都很美好。

直到有一天，我再次对它说：

“请把 Engine324 调到最小运行状态。”

说完，我就去打水了。

回来之后，AI 兴奋地告诉我：

“看起来 Engine324 已经不再需要了，我帮你把它从资源里删除了。”

听到这句话，我心中顿时涌出了很多小可爱。

接下来，我只能一条一条地查询底层资源的配置，再手工把 Engine324 恢复回来。

好在当时这个 Engine 正处于空闲期，没有造成更大的影响。否则，后果可能就不是手工恢复配置这么简单了。

这件事让我意识到：问题不只是 AI 有没有正确理解需求，而是**当 AI 理解错了、判断错了，甚至执行错了，我们有没有办法限制它的操作范围？**

这也是我在 AIDevTeam 中开始考虑代码执行隔离的原因。

## 一、从代码生成到代码执行

传统的 AI Coding 通常是 AI 生成代码，由开发者检查后决定是否执行。代码生成与代码执行之间，还有一道由人把关的环节。

但在 AIDevTeam 中，这个过程已经开始自动化了。

Coder 生成代码后，Tester 会直接运行程序并收集测试结果。如果测试失败，工作流还会将反馈交给 Coder，进入下一轮修改。

这意味着，模型生成的代码不再只是等待人类检查的文本，而是可能被自动执行的程序。

而模型每次生成的代码，**都不能保证是正确、安全的。**

即使用户提出的是一个普通的开发需求，模型也可能生成存在缺陷的代码，或者执行超出预期的操作。

如果这些代码直接运行在宿主机上，并继承了当前进程的权限，就可能读取敏感文件、访问环境变量中的 API Key、修改工作空间之外的文件，甚至在网络权限允许的情况下向外部发送数据。

更麻烦的是，有些操作一旦执行，就很难恢复。即使 Reviewer 随后发现了问题，也无法保证能够撤销已经产生的影响。

因此，当 AI 开发团队具备自动执行代码的能力后，我们就不能只关注代码能否正确运行，还需要考虑：

- 代码可以访问哪些资源？
- 代码可以执行哪些操作？
- 当 AI 出错时，系统能否把影响限制在一个可控的范围内？

## 二、给代码执行建立安全边界

我的初步思路是：**不信任模型生成的代码，而是限制它能够造成的影响。**

第一步，就是将原本集中在一个进程里的工作流拆分成多个容器，让不同的 Agent 运行在相对独立的环境中，并为它们配置不同的访问权限。

### 1. 代码执行隔离

首先，不要直接在用户的环境中运行生成的代码。

可以将代码放到独立的执行环境中运行，并限制它能够访问的文件、网络和系统资源。

在当前版本中，我选择使用容器作为第一步。

不过，需要说明的是，容器化并不自动等于完整的安全沙箱。它提供了隔离的基础，但实际的安全程度仍取决于容器的权限、挂载目录、网络配置以及运行时限制等。

### 2. 最小权限原则

这次改造中，我没有简单的将整个 AIDevTeam 打包到一个镜像里，而是将系统拆分成 Orchestrator、Planner、Coder、Tester 和 Reviewer 几个服务，分别运行在独立的容器中。

这样做的一个好处是：可以根据不同组件的职责，配置不同的访问权限。

例如：Planner和Review，原则上不需要修改工作空间，所以就显示的取消这两个容器对工作空间的写权限

通过这种方式，我们可以先从职责明确、容易实施的权限限制开始，逐步缩小各个组件能够访问的资源范围。

### 3. 仍然需要解决的问题

经过这次改造，我至少可以先把一部分风险控制在容器边界内，不再让所有组件都直接运行在同一个环境中。

但这并不意味着安全问题已经解决。

例如，网络访问、Token 管理、外部 API 调用，以及容器之间可以访问哪些服务，都还需要更细粒度的控制。

**容器化建立的是第一道边界，而不是所有安全问题的最终答案。**

接下来，先看看为了实现这一步，我对 AIDevTeam 的架构做了哪些具体改动。

## 三、具体改动

### 1. 拆分系统拓扑

新版本将系统拆分为五个容器：

- **Orchestrator**：负责维护LangGraph上下文，接收用户需求、启动任务，并协调各个 Agent 的执行。
- **Planner**：负责分析需求并生成开发计划。
- **Coder**：负责根据计划生成和修改代码。
- **Tester**：负责运行生成的代码，并返回测试结果。
- **Reviewer**：负责检查代码和测试结果，判断是否需要继续修改。

其中，Planner、Coder、Tester 和 Reviewer 分别对应 LangGraph 中的四种 Agent 节点，公用大部分代码，只在启动参数和Agent的行为上有所区别；Orchestrator 则负责驱动整个工作流。

为了让不同容器能够协同工作，我采用简单的 HTTP 分装服务间通信，并使用统一的 `run_id` （时间戳+4位随机字符）标识一次开发任务， `job_id` （run_id + agent_role + iteration）标识一次Agent的执行，方便各个组件关联同一个任务，同时也方便日志跟踪不同node的执行过程。

### 2. 统一服务通信与执行接口

为了避免每个 Agent 都重复实现请求处理和数据转换逻辑，我增加了 `rpc` 模块，用于统一处理服务间通信，以及请求和响应对象的序列化、反序列化。

同时，我增加了 `node_server` 模块，将 Agent 包装成 HTTP 服务，并提供两个主要接口：

- `POST /run`：接收 Orchestrator 发来的执行请求。
- `GET /jobs/{job_id}`：根据任务 ID 查询执行结果。

这样，Orchestrator 不需要了解每个 Agent 内部是如何实现的，只需要按照统一的接口提交任务、获取结果即可。

这也为后续增加新的 Agent 或调整内部实现留出了空间。

### 3. 约定 Agent 的组织方式

随着 Agent 数量增加，我也为项目建立了几条简单的约定：

1. 所有 Agent 的实现都放在 `./agents` 目录中。
2. Agent 的服务地址使用对应的 Agent 名称，便于容器之间相互访问。
3. LangGraph 中的节点名称与 Agent 名称保持一致。

这些约定并不复杂，但可以减少不同模块之间的命名差异，也让后续新增 Agent 时有一个统一的实现方式。

## 四、测试一下

架构调整完成后，还是用一个实际任务验证整个流程。

### 1. 启动服务

先启动四个 Agent 服务：

```bash
podman-compose up -d --build planner coder tester reviewer
```

然后启动 Orchestrator，提交一个新的开发需求：

```bash
podman-compose run --rm orchestrator run "写一个把md文件转html的命令行工具"
```

这版的演进没有对AIDevTeam的能力进行扩充，但是用容器方式的重新组织了整个系统的运行的方式，使系统运行有了基础的安全边界，

同时对功能模块进行了切分，方便后续的演化。

### 2. 查看执行过程

可以通过容器日志观察任务的执行情况：

```bash
podman compose logs -f orchestrator planner coder tester reviewer | grep -v 'GET '
```

···
Mac> AIDevTeam$ podman compose logs --follow orchestrator planner coder tester reviewer 
>>>> Executing external compose provider "/opt/homebrew/bin/podman-compose". Please see podman-compose(1) for how to disable this message. <<<<

[orchestrator] | Error: no container with name or ID "aidevteam_orchestrator_1" found: no such container
[coder]        | INFO:     Started server process [1]
[coder]        | INFO:     Waiting for application startup.
[coder]        | INFO:     Application startup complete.
[coder]        | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[reviewer]     | INFO:     Started server process [1]
[reviewer]     | INFO:     Waiting for application startup.
[reviewer]     | INFO:     Application startup complete.
[reviewer]     | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[tester]       | INFO:     Started server process [1]
[tester]       | INFO:     Waiting for application startup.
[tester]       | INFO:     Application startup complete.
[tester]       | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[planner]      | INFO:     Started server process [1]
[planner]      | INFO:     Waiting for application startup.
[planner]      | INFO:     Application startup complete.
[planner]      | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[planner]      | INFO:     10.89.1.6:39970 - "POST /run HTTP/1.1" 202 Accepted
[planner]      | ▶ Planner drafting the build plan…
[planner]      |   app: md2html — 一个纯标准库实现的 Markdown 转 HTML 命令行工具。读取 .md 
[planner]      | 文件，解析标题、段落、有序/无序列表、代码块、行内代码、引用、分隔线、粗体/斜体、
[planner]      | 链接和图片等常见语法，输出 HTML。支持指定输出路径、打印到标准输出、生成完整独立 
[planner]      | HTML 文档（含 doctype 
[planner]      | 与标题）。核心转换逻辑与命令行参数解析分离，便于测试直接导入调用。
[planner]      |   file: app.py — CLI 入口：用 argparse 定义参数（输入 md 文件、-o/--output 
[planner]      | 输出路径、--stdout 打印到终端、--standalone 生成完整 HTML 文档、--title 
[planner]      | 页面标题），读取文件后调用 md2html 的转换函数并写出结果；提供 main() 入口。
[planner]      |   file: md2html.py — 纯逻辑模块：实现轻量 Markdown 解析器。暴露 
[planner]      | markdown_to_html(text, standalone=False, title=None) -> str、render_inline(text)
[planner]      | -> str（行内元素：粗体/斜体/行内代码/链接/图片/HTML 
[planner]      | 转义）以及按行解析块级元素（标题、列表、代码块、引用、分隔线、段落）的辅助函数，
[planner]      | 不涉及任何 I/O 与参数解析。
[coder]        | INFO:     10.89.1.6:41948 - "POST /run HTTP/1.1" 202 Accepted
[coder]        | ▶ Coder implementing the plan…
[coder]        |   wrote md2html.py (14094 bytes)
[coder]        |   notes: Design notes for the team:
[coder]        | 
[coder]        | md2html.py (pure logic, no I/O):
[coder]        | - markdown_to_html(text, standalone=False, title=None) joins parse_blocks() 
[coder]        | output; standalone mode wraps it in a full HTML5 document. Title resolution 
[coder]        | order: explicit --title > first ATX heading > "Untitled".
[coder]        | - render_inline() pipeline: stash inline code spans behind NUL placeholders -> 
[coder]        | HTML-escape the rest -> images -> links -> bold (** then __) -> italic (* then 
[coder]        | _) -> restore code spans as <code>. Attribute values (href/src/title) are 
[coder]        | re-escaped with quote=True when inserted, so URLs cannot break out of 
[coder]        | attributes; main text is escaped with quote=False so link titles using double 
[coder]        | quotes still parse.
[coder]        | - Emphasis regexes use flanking rules (content must start/end with non-space; 
[coder]        | underscore forms additionally require non-word boundaries), so "5 * 3 * 2", 
[coder]        | "some_file_name" and URLs with underscores are not mangled.
[coder]        | - parse_blocks() is a line-based state machine: fenced code (``` / ~~~, optional
[coder]        | language -> class="language-..."), ATX headings (trailing #s stripped), 
[coder]        | horizontal rules, blockquotes (markers stripped, inner text parsed recursively 
[coder]        | -> nesting works), lists (type locked by first line; indented lines continue the
[coder]        | previous item), paragraphs (end at blank line or any block-starting line). 
[coder]        | \r\n/\r normalised.
[coder]        | - Importable helpers for tests: is_blank, is_heading, is_horizontal_rule, 
[coder]        | is_unordered_item, is_ordered_item, is_blockquote, fence_marker, starts_block, 
[coder]        | parse_blocks, render_inline, markdown_to_html.
[coder]        | 
[coder]        | app.py (I/O + argparse only):
[coder]        | - Positional input file; -o/--output (default: input name with .html, also 
[coder]        | handles .markdown/.mdown); --stdout prints instead of writing; --standalone 
[coder]        | emits a full document; --title overrides the page title.
[coder]        | - Files are read as utf-8-sig (strips BOM) and written as utf-8; a trailing 
[coder]        | newline is ensured. Read/write errors go through parser.error (exit code 2). The
[coder]        | "wrote <path>" status line goes to stderr so stdout stays clean for piping; 
[coder]        | --stdout takes precedence over -o.
[coder]        | - main(argv=None) returns an exit code and is callable with an argument list for
[coder]        | tests; build_parser(), default_output_path(), read_text(), write_text() are 
[coder]        | separate for the same reason.
[coder]        | 
[coder]        | Known lightweight-converter limitations (intentional): no setext headings, no 
[coder]        | indented (4-space) code blocks, no nested lists (an indented nested list becomes
[coder]        | continuation text of the parent item), no reference-style links, no autolinks.
[tester]       | INFO:     10.89.1.6:51514 - "POST /run HTTP/1.1" 202 Accepted
[tester]       | ▶ Tester writing and running tests…
[tester]       |   wrote test_md2html_inline.py (4153 chars)
[tester]       |   wrote test_md2html_blocks.py (9971 chars)
[tester]       |   wrote test_md2html_document.py (3064 chars)
[tester]       |   ✘ pytest exit 1: tests failed
[reviewer]     | INFO:     10.89.1.6:46258 - "POST /run HTTP/1.1" 202 Accepted
[reviewer]     | ▶ Reviewer evaluating the submission…
[reviewer]     |   ✘ revisions requested:
[reviewer]     |     - 缺少 CLI 入口，工具目前无法按计划运行。计划明确要求：读取 .md 文件、支持 
[reviewer]     | -o/--output 指定输出路径、默认打印到 stdout、--standalone 生成完整独立文档（含 
[reviewer]     | doctype 与标题，可选 --title），且参数解析与转换核心分离。提交里只有 
[reviewer]     | md2html.py（其 docstring 也自述『不含命令行解析』），但对应的 CLI 
[reviewer]     | 模块没有交付。请补一个 cli.py（argparse + 
[reviewer]     | main()：位置参数输入路径、-o/--output、--standalone、--title；UTF-8 
[reviewer]     | 读写；输入文件不存在/不可读时输出清晰错误并以非零码退出），并实际跑通一次（stdou
[reviewer]     | t 与 -o 两种方式各验证一遍）。
[reviewer]     |     - 修复失败的测试 test_md2html_blocks.py:146 
[reviewer]     | (test_code_content_is_escaped)：是测试期望值写错了，实现是对的。围栏代码块内容必
[reviewer]     | 须整体 HTML 转义，md2html.py 输出的 
[reviewer]     | '<pre><code>&lt;a&gt;&amp;&lt;/a&gt;</code></pre>' 才是正确行为；期望串里 '</a>'
[reviewer]     | 漏了转义。请把期望改为 
[reviewer]     | '<pre><code>&lt;a&gt;&amp;&lt;/a&gt;</code></pre>'。不要为了迎合当前期望去削弱 
[reviewer]     | md2html.py 的转义逻辑。
[coder]        | ▶ Coder revision 2/5…
[coder]        | INFO:     10.89.1.6:46254 - "POST /run HTTP/1.1" 202 Accepted
[coder]        |   wrote md2html.py (13023 bytes)
[coder]        |   wrote cli.py (3837 bytes)
[coder]        |   wrote app.py (514 bytes)
[coder]        |   wrote test_md2html_blocks.py (9270 bytes)
[coder]        |   wrote test_md2html_inline.py (3773 bytes)
[coder]        |   wrote test_cli.py (3772 bytes)
[coder]        |   notes: 按 reviewer 意见修复：
[coder]        | 
[coder]        | 1. 补齐 CLI（此前缺失）：新增 cli.py —— argparse + main()，含位置参数 
[coder]        | input、-o/--output、--stdout（计划要求，可强制覆盖 
[coder]        | -o）、--standalone、--title；UTF-8 读写；输入文件不存在/不可读/非 UTF-8 时向 
[coder]        | stderr 输出清晰错误并返回退出码 1（argparse 用法错误为 2）。app.py 
[coder]        | 保留为计划规定的启动入口（python app.py ...），委托 
[coder]        | cli.main()，从而参数解析（cli.py）与转换核心（md2html.py，纯逻辑、无 
[coder]        | I/O）严格分离。
[coder]        | 2. 修复 test_md2html_blocks.py 的 test_code_content_is_escaped：期望值改为 
[coder]        | '<pre><code>&lt;a&gt;&amp;&lt;/a&gt;</code></pre>'（闭合标签 </a> 
[coder]        | 同样转义）；未削弱 md2html.py 的转义逻辑——代码块内容仍整体 html.escape。
[coder]        | 3. 已实际跑通两种输出方式：
[coder]        |    - stdout：`python app.py demo.md`（demo.md 为 "# 标题\n\n正文 **粗体**。"）→ 
[coder]        | stdout 输出 `<h1>标题</h1>` 与 `<p>正文 <strong>粗体</strong>。</p>`，退出码 0；
[coder]        |    - -o：`python app.py demo.md -o demo.html` → 写出 
[coder]        | demo.html（内容同上，UTF-8），stdout 保持干净，stderr 提示 "md2html: 已写入 
[coder]        | demo.html"；
[coder]        |    - `python -m unittest discover -v`（blocks / inline / cli 
[coder]        | 三个测试模块）全部通过，其中 test_cli.py 以进程内与 subprocess 两种方式再次验证 
[coder]        | stdout 与 -o 路径。
[coder]        | 其他说明：md2html.py 为纯函数模块（markdown_to_html / render_inline / 
[coder]        | parse_blocks 
[coder]        | 及块级渲染辅助函数），支持标题、段落、有序/无序列表（含缩进续行）、围栏代码块（`
[coder]        | `` 与 ~~~，语言输出为 
[coder]        | class="language-xxx"）、引用（递归解析）、分隔线、粗体/斜体/粗斜体、行内代码、链
[coder]        | 接、图片，其余字符统一 HTML 转义；standalone 模式输出含 doctype 与 <title> 
[coder]        | 的完整文档，标题缺省用 DEFAULT_TITLE，可用 --title 覆盖且会做 HTML 转义。
[tester]       | ▶ Tester writing and running tests…
[tester]       | INFO:     10.89.1.6:43564 - "POST /run HTTP/1.1" 202 Accepted
[tester]       |   removed stale test_md2html_document.py
[tester]       |   removed stale test_md2html_inline.py
[tester]       |   removed stale test_cli.py
[tester]       |   removed stale test_md2html_blocks.py
[tester]       |   wrote test_md2html_inline.py (4286 chars)
[tester]       |   wrote test_md2html_blocks.py (7801 chars)
[tester]       |   wrote test_md2html_helpers.py (6642 chars)
[tester]       |   wrote test_cli.py (6103 chars)
[tester]       |   ✘ pytest exit 1: tests failed
[reviewer]     | ▶ Reviewer evaluating the submission…
[reviewer]     | INFO:     10.89.1.6:42832 - "POST /run HTTP/1.1" 202 Accepted
[reviewer]     |   ✘ revisions requested:
[reviewer]     |     - 修复 md2html.py:190 `is_closing_fence` 的 KeyError（导致全部 10 
[reviewer]     | 个围栏/代码块失败：test_backtick_fence、test_code_content_is_escaped、test_code_
[reviewer]     | block_with_language、test_code_info_uses_first_word_only、test_tilde_fence、test
[reviewer]     | _different_fence_char_does_not_close_block、test_unclosed_fence_runs_to_end_of_i
[reviewer]     | nput、test_inline_syntax_not_processed_inside_code 及 test_md2html_helpers.py 中
[reviewer]     | 2 个）：`r"^ {0,3}{0}{{3,}}[ \t]*$".format(...)` 里的字面量 `{0,3}` 被 
[reviewer]     | str.format 当成替换字段。改为转义花括号 `r"^ {{0,3}}{0}{{3,}}[ 
[reviewer]     | \t]*$".format(re.escape(fence_char))`，或直接拼接 `r"^ {0,3}" + 
[reviewer]     | re.escape(fence_char) + r"{3,}[ \t]*$"`。
[reviewer]     |     - 修复剩余 6 
[reviewer]     | 个失败：test_md2html_blocks.py::test_heading_levels[1-6]（AssertionError，报告截
[reviewer]     | 断看不到期望值）。运行 `pytest test_md2html_blocks.py -k heading_levels -vv` 
[reviewer]     | 查看实际输出与期望的差异，据此修正 render_heading/parse_blocks，使 1~6 
[reviewer]     | 级标题全部通过。
[reviewer]     |     - 修复后重跑完整测试套件确认 0 失败（当前 153 通过 / 16 失败），CLI 
[reviewer]     | 冒烟（app.py --help）已通过，无需改动入口。
[coder]        | ▶ Coder revision 3/5…
[coder]        | INFO:     10.89.1.6:57728 - "POST /run HTTP/1.1" 202 Accepted
[coder]        |   wrote md2html.py (13703 bytes)
[coder]        |   wrote app.py (2377 bytes)
[coder]        |   notes: 修复了评审指出的全部 16 个失败，其余逻辑保持不变。
[coder]        | 
[coder]        | 1. is_closing_fence KeyError（修复 10 个围栏/代码块测试 + 2 个 helper 测试）：
[coder]        |    - 原因：用 str.format 构造正则时，字面量 `{0,3}` 被当成替换字段（KeyError: 
[coder]        | '0,3'）。
[coder]        |    - 修复：改为纯字符串拼接 `r"^ {0,3}" + re.escape(fence_char) + r"{3,}[ 
[coder]        | \t]*$"`（评审建议的第二种方式），不再对含字面量花括号的正则调用 
[coder]        | format。模块内所有正则均为普通字面量，动态拼接仅此一处。
[coder]        |    - 逐项验证：test_backtick_fence / 
[coder]        | test_tilde_fence（开栏与闭栏匹配同一字符）、test_code_block_with_language 与 
[coder]        | test_code_info_uses_first_word_only（info 取第一个单词，输出 
[coder]        | class="language-xxx"）、test_code_content_is_escaped（内容 
[coder]        | html.escape，不做行内处理）、test_different_fence_char_does_not_close_block（~~~
[coder]        | 不会关闭 ``` 
[coder]        | 代码块，作为内容保留）、test_unclosed_fence_runs_to_end_of_input（无闭栏时消费到
[coder]        | 末尾）、test_inline_syntax_not_processed_inside_code（**bold**、(x) 
[coder]        | 在代码块内原样转义输出）、test_md2html_helpers.py 中直接调用 
[coder]        | is_closing_fence(line, fence_char) 的 2 个用例不再抛 KeyError。
[coder]        | 
[coder]        | 2. test_heading_levels[1-6]（修复 6 个标题测试）：
[coder]        |    - parse_heading 使用 `^ {0,3}(#{1,6})(?:\s+(.*))?$`，级别 = 
[coder]        | 井号个数（1~6），要求井号后必须有空白；render_heading 输出 
[coder]        | `<h{level}>{render_inline(text)}</h{level}>`。`#`→`<h1>` … `######`→`<h6>` 
[coder]        | 全部通过；`#######`（7 个）按段落处理，`## T ##` 会剥离结尾闭合井号。
[coder]        | 
[coder]        | 3. 其余保持/达到正确行为：段落、有序/无序列表（<ul>/<ol> + 
[coder]        | <li>）、引用（递归解析，支持嵌套）、分隔线（---/***）、行内代码优先提取（其内容
[coder]        | 只转义、不做行内替换）、图片先于链接匹配、粗体/斜体（含 ___、*** 
[coder]        | 及下划线词边界保护）、HTML 转义（&<>，属性额外转义引号）、standalone 
[coder]        | 完整文档（doctype/head/title/body）。核心逻辑与 argparse 完全分离，md2html.py 
[coder]        | 无任何 I/O；app.py 入口未改动行为（--help 冒烟通过，-o 
[coder]        | 写文件、--stdout/--standalone/--title 均可用）。
[tester]       | ▶ Tester writing and running tests…
[tester]       | INFO:     10.89.1.6:60210 - "POST /run HTTP/1.1" 202 Accepted
[tester]       |   removed stale test_md2html_helpers.py
[tester]       |   removed stale test_md2html_inline.py
[tester]       |   removed stale test_cli.py
[tester]       |   removed stale test_md2html_blocks.py
[tester]       |   wrote test_md2html_helpers.py (7435 chars)
[tester]       |   ✔ pytest exit 0: all tests passed
[reviewer]     | ▶ Reviewer evaluating the submission…
[reviewer]     | INFO:     10.89.1.6:36080 - "POST /run HTTP/1.1" 202 Accepted
[reviewer]     |   ✔ approved
···

从日志中可以看到，各个 Agent 会按照工作流依次执行。

对于稍微复杂一些的需求，Tester 和 Reviewer 已经能够发现部分问题，并将反馈交给 Coder 修改，随后继续进入测试和审查流程。

### 3. 测试结果

![AIDevTeam 实际执行](./docAsset/example.png)

从这次测试来看，这版容器化改造已经基本跑通：Orchestrator 可以驱动不同的 Agent 完成任务，Agent 之间能够通过 HTTP 通信，Tester 和 Reviewer 也能够参与后续的迭代修改。

当然，这次验证主要证明了**多容器架构和工作流能够正常协同运行**，并不代表所有安全边界都已经经过充分验证。

下一步，我还需要针对文件访问、网络权限、敏感 Token 和资源限制等方面，逐一补上更严格的控制与测试。

至少在当前阶段，我可以安心的去打杯水了