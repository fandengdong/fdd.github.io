---
title: "论文阅读：Attention Is All You Need"
date: 2025-11-06T18:00:00+08:00
draft: false
math: true
---

本文简要记录对经典论文《Attention Is All You Need》的理解。

## 数学公式示例

Transformer 中的缩放点积注意力公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

行内公式：$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$。

## 表格示例

| Layer | Output Size | Params |
|-------|-------------|--------|
| Embedding | 512 | 1M |
| Encoder | 512 | 60M |
| Decoder | 512 | 60M |

## 插入图片

将图片放在同目录下，同目录下面必须有一个index.md文件，然后引用：

<!-- ![Transformer 架构图](transformer.png)  -->
![Transformer 架构图](transformer.png) 

> 提示：确保文件名称为index.md,_index.md文件也不行。
