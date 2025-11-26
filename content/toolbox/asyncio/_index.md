---
title: "python中的异步框架 - asyncio"
description: "学习asyncio的概念和用法笔记"
---

asyncio 是 Python 标准库中的异步 I/O 框架，用于编写高并发、事件驱动的程序。它基于 协程（coroutine） 和 事件循环（event loop），特别适合处理大量 I/O 密集型任务（如网络请求、文件读写、数据库访问等）。

下面是一个简单的例子，展示如何使用 asyncio 创建一个异步任务：
```python
import asyncio
import time

async def task1():
    print(f"[{time.time():.2f}] 开始做任务1")
    await asyncio.sleep(2)
    print(f"[{time.time():.2f}] 任务1完成")

async def task2():
    print(f"[{time.time():.2f}] 开始做任务2")
    await asyncio.sleep(2)
    print(f"[{time.time():.2f}] 任务2完成")

async def main():
    print(f"[{time.time():.2f}] main: 创建 t1")
    t1 = asyncio.create_task(task1()) # 任务1准备就绪
    
    print(f"[{time.time():.2f}] main: 创建 t2")
    t2 = asyncio.create_task(task2()) # 任务2准备就绪
    
    print(f"[{time.time():.2f}] main: 开始 await t1")
    await t1 # 任务1和任务2开始执行，等待任务1完成
    print(f"[{time.time():.2f}] main: await t1 完成")
    
    await t2 # 等待任务2完成
    print(f"[{time.time():.2f}] main: 全部结束")

# 更简洁main写法
# async def main():
#     await asyncio.gather(task1(), task2())

asyncio.run(main())
```

这段代码输出为：
```bash
[1764148547.33] main: 创建 t1
[1764148547.33] main: 创建 t2
[1764148547.33] main: 开始 await t1
[1764148547.33] 开始做任务1
[1764148547.33] 开始做任务2
[1764148549.33] 任务1完成
[1764148549.33] 任务2完成
[1764148549.33] main: await t1 完成
[1764148549.33] main: 全部结束
```

> 💡 asyncio 让你在单线程中高效“同时”处理成百上千个 I/O 操作，核心秘诀是——在等待时去做别的事。
