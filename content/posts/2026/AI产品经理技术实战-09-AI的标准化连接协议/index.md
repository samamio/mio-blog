---
title: "MCP——AI的标准化连接协议"
date: 2026-06-18T00:00:00+08:00
description: "前几章我们学了： - Function Calling：让模型调用工具 - RAG：让模型获取知识 - Agent：让模型自主规划 - LangGraph：构建有状态的Agent工作流"
tags: [AI, MCP, 协议, 标准化, 连接]
draft: false
---

# 第9章 MCP——AI的标准化连接协议


---

## 一、背景：为什么需要MCP？

### 1.1 AI时代的"巴别塔"问题

想象一下：

> 你有10个AI工具（CRM、ERP、工单系统、邮件、日历、文档、数据库、搜索、通知、分析）。
> 
> 每个工具的API都不一样：
> - CRM用REST API
> - ERP用GraphQL
> - 工单系统用gRPC
> - 邮件系统用SMTP
> - ...
> 
> 要让AI跟这10个工具对话，你需要写10套不同的适配器。
> 
> 如果再加一个工具，又要写一套。
> 
> 如果换一个AI模型，又要重写所有适配器。

这就是AI时代的"巴别塔"问题——每个工具都有自己的"语言"，AI需要学会每种语言才能对话。

### 1.2 MCP的解决方案：一个协议，所有工具

MCP（Model Context Protocol）的目标很简单：

> **定义一个标准协议，让AI模型和外部工具/数据源之间的通信标准化。**

有了MCP之后：

```
AI模型 ←MCP协议→ MCP Server ←各种API→ 外部系统
```

- MCP Server负责"翻译"：把外部系统的API转换成MCP标准格式
- AI模型只需要"会说"MCP：不需要关心外部系统用什么API
- 新增外部系统：只需要写一个新的MCP Server，AI模型不需要改

### 1.3 MCP的诞生

MCP由Anthropic（Claude的开发商）在2024年提出，并开源为一个标准协议。

**为什么是Anthropic？**

因为Anthropic发现，他们在构建Claude产品时，需要对接大量的外部工具和数据来源。每次对接都要写新的适配器，效率很低。于是他们决定把这个适配层标准化。

**MCP的核心洞察：**

> AI模型和外部世界之间的接口，应该标准化。就像USB接口标准化了电脑和外设的连接一样，MCP标准化了AI和外部世界的连接。

### 1.4 MCP vs Function Calling：区别是什么？

很多人会混淆MCP和Function Calling。它们的关系是：

**Function Calling是"语法"，MCP是"协议"。**

- Function Calling：模型决定调用哪个函数、传什么参数
- MCP：定义了函数（工具）怎么被发现、怎么被调用、怎么返回结果

换句话说：

- 没有MCP，你也可以用Function Calling——但每次都要手写适配器
- 有了MCP，Function Calling变成标准化的——模型通过MCP协议自动发现和使用工具

**类比：**

- Function Calling = HTTP请求（具体的操作）
- MCP = HTTP协议（规定了请求和响应的格式）

---

## 二、核心原理：MCP的三个核心抽象

### 2.1 Resource：资源

**Resource是MCP的第一个核心抽象。**

Resource代表"数据"——模型可以读取的外部数据源。

**类比：**

- 数据库表 → Resource
- 文件 → Resource
- API返回的JSON → Resource
- 网页内容 → Resource

**Resource的特点：**

- 只读的（模型只能读取，不能修改）
- 有URI标识的（每个Resource有一个唯一的地址）
- 有MIME类型的（文本、JSON、HTML等）

**示例：**

```
URI: mcp://crm/contacts/12345
MIME Type: application/json
Content: {"name": "张三", "company": "ABC科技", "phone": "138xxxx"}
```

### 2.2 Prompt：提示

**Prompt是MCP的第二个核心抽象。**

Prompt代表"预定义的对话模板"——模型可以使用的标准化提示词。

**类比：**

- 函数模板 → Prompt
- 预设的对话开场白 → Prompt
- 标准化的任务描述 → Prompt

**Prompt的特点：**

- 预定义的输入参数
- 预定义的对话模板
- 可以被模型"调用"

**示例：**

```
Name: summarize_document
Description: 总结一份文档的核心内容
Arguments:
  - document_uri: 文档的URI
  - max_length: 最大字数（可选，默认200）

Template:
"请总结以下文档的核心内容，不超过{max_length}字：

{document_content}"
```

### 2.3 Tool：工具

**Tool是MCP的第三个核心抽象。**

Tool代表"可执行的操作"——模型可以调用的外部功能。

**类比：**

- API方法 → Tool
- 函数 → Tool
- 可执行的脚本 → Tool

**Tool的特点：**

- 可读写的（模型可以调用并修改外部状态）
- 有输入参数的
- 有返回结果的

**示例：**

```
Name: create_ticket
Description: 创建一个新的客服工单
Arguments:
  - user_id: 用户ID
  - category: 工单分类
  - description: 工单描述
  - priority: 优先级（可选，默认"中"）

Return:
  - ticket_id: 创建的工单ID
  - status: 创建状态
```

### 2.4 三者的关系

```
┌─────────────────────────────────────────┐
│           MCP的三个抽象                  │
│                                         │
│  Resource（数据）                        │
│  ├── 读取外部数据                        │
│  ├── 比如：CRM联系人信息                 │
│  ├── 比如：工单详情                      │
│  └── 特点：只读                          │
│                                         │
│  Prompt（对话模板）                       │
│  ├── 预定义的对话模板                    │
│  ├── 比如：文档总结模板                  │
│  ├── 比如：代码审查模板                  │
│  └── 特点：可复用                        │
│                                         │
│  Tool（可执行操作）                       │
│  ├── 执行外部操作                        │
│  ├── 比如：创建工单                      │
│  ├── 比如：发送邮件                      │
│  └── 特点：可读写                        │
│                                         │
│  关系：                                  │
│  Tool操作数据（Resource）                │
│  Prompt引导Tool的使用                    │
└─────────────────────────────────────────┘
```

---

## 三、企业应用：MCP的架构

### 3.1 MCP的架构模型

MCP采用**Server-Client**架构：

```
┌───────────────────────────────────────────────┐
│              MCP Client                        │
│          (AI模型 / 应用)                        │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │            MCP Protocol                 │  │
│  └─────────────────────────────────────────┘  │
└──────────────────┬────────────────────────────┘
                   │ MCP协议
                   │ (JSON-RPC over stdio / HTTP)
                   ▼
┌───────────────────────────────────────────────┐
│              MCP Server                        │
│          (工具/数据提供者)                       │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Resource  │  │  Prompt  │  │  Tool    │   │
│  │Provider  │  │ Provider │  │ Provider │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │          │
│       ▼             ▼             ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  CRM     │  │  模板库   │  │  工单系统 │   │
│  │  API     │  │          │  │  API     │   │
│  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────────────────────────┘
```

**Client（客户端）：** 发起请求的一方。通常是AI模型或AI应用。

**Server（服务端）：** 提供能力的一方。通常是外部系统（数据库、API、文件系统等）的封装。

### 3.2 通信协议

MCP支持两种通信方式：

**方式一：stdio（标准输入输出）**

```
AI应用 ←stdin/stdout→ MCP Server（本地进程）
```

- 适用于本地MCP Server
- 延迟低
- 适合开发和本地工具

**方式二：HTTP（网络）**

```
AI应用 ←HTTP/SSE→ MCP Server（远程服务）
```

- 适用于远程MCP Server
- 可以跨网络
- 适合生产环境

### 3.3 本地MCP vs 远程MCP

| 维度 | 本地MCP | 远程MCP |
|------|---------|---------|
| 部署位置 | 运行在用户本地机器上 | 运行在远程服务器上 |
| 延迟 | 低（进程间通信） | 高（网络通信） |
| 数据安全性 | 数据不出本地 | 数据需要发送到远程 |
| 适用场景 | 个人工具、开发环境 | 企业级服务、多用户场景 |
| 举例 | 本地文件系统MCP Server | CRM云端MCP Server |

**产品经理的选择：**

- 面向个人用户的产品 → 本地MCP（隐私性好、延迟低）
- 面向企业用户的产品 → 远程MCP（集中管理、数据可审计）

### 3.4 MCP生态现状

截至2024年底，MCP生态正在快速发展：

**官方Server：**

- 文件系统MCP Server：让AI读取本地文件
- GitHub MCP Server：让AI访问GitHub仓库
- Google Drive MCP Server：让AI读写Google Drive
- PostgreSQL MCP Server：让AI查询数据库

**社区Server：**

- Salesforce MCP Server
- Slack MCP Server
- Notion MCP Server
- Jira MCP Server

**AI应用对MCP的支持：**

- Claude Desktop：原生支持MCP
- Cursor：支持MCP
- Windsurf：支持MCP
- 越来越多的AI工具正在加入MCP支持

---

## 四、产品经理需要掌握什么？

### 4.1 MCP视角下的产品设计

理解了MCP之后，产品经理在設計AI产品时可以有新的思路：

**设计原则一：模块化**

每个外部系统封装成一个MCP Server。新增系统不需要改现有代码，只需要加一个新的Server。

**设计原则二：即插即用**

AI应用通过MCP协议自动发现可用的Server和能力。不需要硬编码。

**设计原则三：能力共享**

同一个MCP Server可以被多个AI应用共享。比如CRM的MCP Server可以被客服Agent、销售Agent、营销Agent共用。

### 4.2 MCP产品的PRD写法

**模块一：MCP Server清单**

```
| Server名称 | 提供能力 | 类型 | 部署方式 |
|-----------|---------|------|---------|
| CRM Server | 读取客户信息、更新客户状态 | Resource + Tool | 远程 |
| 工单Server | 查询/创建/更新工单 | Resource + Tool | 远程 |
| 知识库Server | 检索知识文档 | Resource | 远程 |
| 邮件Server | 发送邮件 | Tool | 远程 |
| 文件系统Server | 读写本地文件 | Resource + Tool | 本地 |
```

**模块二：MCP通信协议**

```
- 内部系统：stdio（本地进程通信）
- 外部系统：HTTP+SSE（服务器发送事件）
- 认证方式：OAuth 2.0
```

**模块三：能力发现**

```
AI应用启动时：
1. 自动发现可用的MCP Server
2. 获取每个Server提供的Resource、Prompt、Tool列表
3. 缓存能力清单
4. 运行时根据用户请求选择对应的Tool
```

### 4.3 跟工程师沟通的常用话术

**讨论MCP Server设计时：**

> "这个外部系统封装成MCP Server后，能提供哪些Resource、Prompt和Tool？"

**讨论能力发现时：**

> "AI应用怎么知道有哪些MCP Server可用？是启动时自动发现的，还是硬编码配置的？"

**讨论安全时：**

> "MCP Server的认证方式是什么？不同用户的权限怎么隔离？"

---

## 五、研发如何实现：MCP的工程实现

### 5.1 MCP Server的开发流程

```
1. 确定要暴露的能力
   - Resource：哪些数据需要让AI读取？
   - Prompt：哪些对话模板需要预定义？
   - Tool：哪些操作需要让AI执行？

2. 实现Server
   - 定义Resource URI和读取逻辑
   - 定义Prompt模板和参数
   - 定义Tool名称、参数和执行逻辑

3. 注册和发现
   - 启动Server
   - 向MCP注册中心注册
   - AI应用发现Server

4. 运行
   - AI应用通过MCP协议调用Server的能力
   - Server执行操作并返回结果
```

### 5.2 MCP Server代码示例（伪代码）

```python
from mcp import Server, Resource, Tool

# 创建Server
server = Server("customer-service")

# 定义Resource
@server.resource("mcp://tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """获取工单详情"""
    ticket = db.query_ticket(ticket_id)
    return {"uri": f"mcp://tickets/{ticket_id}", "content": ticket}

# 定义Tool
@server.tool("create_ticket")
async def create_ticket(user_id: str, category: str, description: str):
    """创建新工单"""
    ticket = db.create_ticket(user_id=user_id, category=category, description=description)
    return {"ticket_id": ticket.id, "status": "created"}

# 定义Prompt
@server.prompt("summarize_ticket")
async def summarize_ticket(ticket_id: str):
    """总结工单"""
    ticket = db.query_ticket(ticket_id)
    return {
        "messages": [
            {"role": "user", "content": f"请总结以下工单：{ticket.description}"}
        ]
    }

# 启动Server
server.run()
```

### 5.3 MCP的安全设计

```
认证：
- OAuth 2.0 / API Key
- 每个MCP Server需要认证才能访问

授权：
- RBAC（基于角色的访问控制）
- 不同用户能看到不同的Resource和Tool

审计：
- 记录所有MCP调用日志
- 包括谁在什么时候调用了什么Tool
```

---

## 六、完整流程图

### MCP架构全景图

```
┌─────────────────────────────────────────────────────┐
│                  AI应用层                            │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 客服Agent │  │ 销售Agent │  │ 营销Agent │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │                │
│       └─────────────┼─────────────┘                │
│                     │                              │
│              ┌──────▼──────┐                       │
│              │  MCP Client  │                       │
│              │  (协议层)     │                       │
│              └──────┬──────┘                       │
└─────────────────────┼─────────────────────────────┘
                      │ MCP协议
            ┌─────────┼─────────┐
            │         │         │
            ▼         ▼         ▼
┌───────────────────────────────────────────────┐
│              MCP Server层                     │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ CRM      │  │ 工单     │  │ 知识库   │   │
│  │ Server   │  │ Server   │  │ Server   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │          │
│       ▼             ▼             ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ CRM API  │  │ 工单API  │  │ 向量DB   │   │
│  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────────────────────────┘
```

### MCP能力发现流程

```
AI应用启动
    │
    ▼
扫描可用的MCP Server
    │
    ▼
连接每个Server
    │
    ▼
获取Server声明的能力
    ├── Resource列表
    ├── Prompt列表
    └── Tool列表
    │
    ▼
缓存能力清单
    │
    ▼
用户请求 → 匹配能力 → 调用对应Server
```

---

## 七、真实案例：客户工单系统的MCP设计

### 7.1 背景

客户工单系统需要对接多个外部系统：

- CRM系统（客户信息）
- 工单系统（工单数据）
- 邮件系统（发送通知）
- 知识库（FAQ和操作手册）
- 内部数据库（用户数据）

### 7.2 MCP Server设计

```
Server 1: CRM MCP Server
  Resource: mcp://crm/customers/{customer_id}
  Resource: mcp://crm/companies/{company_id}
  Tool: update_customer(customer_id, fields)
  Tool: search_customers(query)

Server 2: Ticket MCP Server
  Resource: mcp://tickets/{ticket_id}
  Resource: mcp://tickets/user/{user_id}
  Tool: create_ticket(user_id, category, description)
  Tool: update_ticket(ticket_id, fields)
  Tool: close_ticket(ticket_id, resolution)

Server 3: Knowledge MCP Server
  Resource: mcp://knowledge/{doc_id}
  Resource: mcp://knowledge/search/{query}
  Tool: none (知识库只读)

Server 4: Email MCP Server
  Tool: send_email(to, subject, body)
  Tool: send_ticket_update(ticket_id, message)
```

### 7.3 架构优势

**新增系统时：**

只需要写一个新的MCP Server，不需要改现有的AI应用代码。

**更换AI模型时：**

AI应用只需要支持MCP协议，不需要关心MCP Server用什么技术实现。

**多Agent共享：**

客服Agent、销售Agent、营销Agent都可以使用同一个MCP Server，不需要各自对接一次。

---

## 八、常见误区

### 误区一："MCP是必须的"

**真相**：如果你的产品只需要对接1-2个系统，手写适配器就够了，不需要MCP。MCP的价值在对接多个系统时才体现出来。

**实践方法**：产品初期用Function Calling手写适配器，系统超过10个时再考虑MCP。

### 误区二："MCP是一个新的API"

**真相**：MCP不是一个API，而是一个协议。它规定了AI和外部系统之间通信的格式。

**实践方法**：MCP Server可以用任何语言实现（Python、Go、Node.js），只要遵循MCP协议就行。

### 误区三："MCP只支持JSON-RPC"

**真相**：MCP支持多种传输方式（stdio、HTTP、WebSocket），不限于JSON-RPC。

### 误区四："MCP生态已经很丰富了"

**真相**：MCP还是一个比较新的协议，生态在快速发展但还不够成熟。很多系统还没有官方的MCP Server。

**实践方法**：在产品设计时预留MCP兼容性，但不要完全依赖MCP生态。

### 误区五："MCP能替代所有API集成"

**真相**：MCP解决的是"AI和外部世界的接口"问题，不是所有API集成的问题。传统的系统间集成（如两个后端服务之间的通信）不需要MCP。

### 误区六："MCP很复杂"

**真相**：MCP协议本身很简单——就是定义Resource、Prompt、Tool的格式。实现一个基本的MCP Server并不难。

### 误区七："MCP只跟AI模型有关"

**真相**：MCP的价值在于"标准化"。一旦一个系统有了MCP Server，任何支持MCP的AI应用都可以使用它，不局限于某一个模型。

---

## 九、面试题

### 初级

**Q1：什么是MCP？它解决了什么问题？**

**参考答案：**

MCP（Model Context Protocol）是AI模型和外部世界之间的标准化连接协议。

解决的问题：

1. 每个外部系统的API都不一样，AI需要为每个系统写不同的适配器
2. MCP提供了一个标准协议，让AI可以通过一种方式对接所有系统
3. 新增系统时，只需要写一个新的MCP Server，AI不需要改代码

类比：MCP之于AI，就像USB之于电脑——统一了接口标准。

---

**Q2：MCP的三个核心抽象是什么？**

**参考答案：**

1. Resource：只读的外部数据源（如数据库记录、文件内容）
2. Prompt：预定义的对话模板（如文档总结模板、代码审查模板）
3. Tool：可执行的操作（如创建工单、发送邮件）

---

### 中级

**Q3：MCP Server和MCP Client的关系是什么？**

**参考答案：**

MCP Server提供能力（Resource、Prompt、Tool），MCP Client消费能力。

- Server是"供给方"：封装外部系统，暴露标准接口
- Client是"需求方"：AI模型或AI应用，通过MCP协议调用Server的能力

关系：Client发起请求，Server响应请求。类似HTTP的Client-Server模型，但协议是MCP。

---

**Q4：本地MCP和远程MCP有什么区别？什么时候用哪种？**

**参考答案：**

本地MCP运行在用户本地机器上，通过stdio通信。优点是延迟低、隐私性好。适合个人工具和开发环境。

远程MCP运行在远程服务器上，通过HTTP通信。优点是集中管理、多用户共享。适合企业级服务和生产环境。

选择：面向个人用户用本地MCP，面向企业用户用远程MCP。

---

### 高级

**Q5：你负责一个AI产品，需要对接15个外部系统。你会用MCP吗？为什么？**

**参考答案：**

会。原因：

1. **减少重复开发**：15个系统意味着15套不同的API。用MCP只需要写15个Server，AI应用不需要改。
2. **能力共享**：15个系统的MCP Server可以被多个Agent共享，不需要每个Agent都对接一次。
3. **易于扩展**：未来增加第16个系统时，只需要加一个Server，不需要改现有代码。
4. **模型无关**：如果未来换AI模型，MCP Server不需要重写。

但如果只有1-2个系统，就不需要MCP了——手写适配器更简单。

---

**Q6：MCP的生态现状如何？作为产品经理，你怎么评估MCP的成熟度？**

**参考答案：**

评估维度：

1. **Server数量**：目前约有20-30个官方和社区MCP Server，覆盖常用系统（文件系统、GitHub、Google Drive等）。相比REST API的数百万生态，还很小。
2. **AI应用支持**：Claude Desktop、Cursor、Windsurf等支持MCP，但大多数AI应用还不支持。
3. **标准化程度**：MCP协议还在快速迭代中，未来可能会有 breaking changes。
4. **企业采用**：大型企业对MCP持观望态度，在等生态成熟后再投入。

**我的建议**：

- 短期：在产品设计中预留MCP兼容性，但不完全依赖
- 中期：优先支持核心系统的MCP Server
- 长期：当MCP生态成熟后，全面迁移到MCP架构

---

## 十、本章总结

本章我们学习了MCP的核心概念和应用：

1. **MCP的背景**：AI时代"巴别塔"问题——每个系统API不同，AI需要写很多适配器
2. **MCP的解决方案**：一个标准协议，让AI通过统一接口对接所有系统
3. **三个核心抽象**：Resource（数据）、Prompt（模板）、Tool（操作）
4. **Server-Client架构**：Server提供能力，Client消费能力
5. **本地MCP vs 远程MCP**：延迟、安全性、适用场景的差异
6. **MCP生态现状**：快速发展中，但还不够成熟
7. **客户工单系统MCP设计**：4个Server覆盖CRM、工单、知识库、邮件
8. **MCP的价值**：模块化、即插即用、能力共享

---

## 十一、延伸阅读

### 官方资源

1. **MCP Official Specification** — MCP协议的官方规范文档
2. **MCP GitHub Repository** — MCP的开源实现和社区
3. **MCP Server Registry** — 官方MCP Server列表

### 教程和指南

1. **Anthropic MCP Introduction** — Anthropic官方的MCP介绍
2. **MCP Quick Start Guide** — 快速上手MCP的指南

### 工具

1. **MCP CLI** — 命令行工具，用于测试和调试MCP Server
2. **MCP Inspector** — 可视化MCP Server的工具

### 社区

1. **MCP Discord** — MCP实时讨论
2. **MCP GitHub Discussions** — MCP协议讨论和发展

---

## 下一章预告

第十章我们将进入**多Agent系统**——从单兵作战到团队协作。

你会学到：
- 为什么单Agent走不远——复杂度增长带来的能力稀释
- 多Agent的三种组织模式：Master-Worker、Peer-to-Peer、Swarm
- Agent角色设计——产品经理式的能力画像与接口定义
- Agent间通信协议——共享状态、消息传递、同步机制
- 冲突解决与一致性——当两个Agent给出不同结论怎么办
- 客户工单系统：升级到多Agent架构（接待Agent + 查询Agent + 决策Agent + 质检Agent）

准备好你的团队，第十章会让你从"一个AI"升级到"一群AI"。
