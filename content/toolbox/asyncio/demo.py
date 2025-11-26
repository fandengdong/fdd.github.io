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
    t1 = asyncio.create_task(task1())
    
    print(f"[{time.time():.2f}] main: 创建 t2")
    t2 = asyncio.create_task(task2())
    
    print(f"[{time.time():.2f}] main: 开始 await t1")
    await t1
    print(f"[{time.time():.2f}] main: await t1 完成")
    
    await t2
    print(f"[{time.time():.2f}] main: 全部结束")

asyncio.run(main())
