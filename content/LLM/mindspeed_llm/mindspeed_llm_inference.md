---
title: "使用 Mindspeed-LLM进行推理"
date: 2025-11-14
---

本文记录了 mindspeed-llm 框架提供的推理方法，以 Qwen2.5-7B 模型为例说明推理过程。

## 准备工作

推理模型的权重需要转换为 mcore 格式，方法与微调时相同，参考[这里](https://fandengdong.github.io/llm/mindspeed_llm/mindspeed_llm_finetune/#%E5%87%86%E5%A4%87%E6%9D%83%E9%87%8D)

## 开启推理测试

准备好模型权重后，可以直接使用 mindspeed-llm 提供的推理脚本进行推理测试，脚本位于`examples/mcore/qwen25/generate_qwen25_7b_ptd.sh`

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="/home/fdd/workspace/models/Qwen/Qwen2.5-7B/mcore_tp1_pp1/"
TOKENIZER_PATH="/home/fdd/workspace/models/Qwen/Qwen2.5-7B-Instruct/"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
SEQ_LENGTH=32768

...


torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       ...
```

可以看到对于 7B 模型，采用单卡就可以进行推理。启动方式依然为 torchrun，启动脚本为 inference.py。

## 推理流程

查看inference.py文件：

```python

from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM

def main():
    initialize_megatron(args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()

    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    task_factory(args, model)


if __name__ == "__main__":
    main()
```

可推理流程的关键步骤包括：

环境初始化：调用 initialize_megatron 方法初始化 Megatron 环境，禁用随机数加载和优化器加载功能（推理不需要这些组件）
模型加载：通过 MegatronModuleForCausalLM.from_pretrained 方法初始化模型，其中 MegatronModuleForCausalLM 是专为推理设计的类
任务执行：根据 args.task 参数调用 task_factory 方法执行具体任务。

## 模型结构分析

我们进一步看下推理模型的结构，即model_provider内容：

```python
def model_provider(pre_process=True, post_process=True):
    ...
    if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    ...
    return model
```

这里关键关注GPTModelInfer的定义：

```python

class GPTModelInfer(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_model = MegatronModuleForCausalLM()

    def generate(self, input_ids=None, **kwargs):
        return self.infer_model.generate(input_ids=input_ids, **kwargs)

```

GPTModelInfer 模型继承自 GPTModel，增加了 MegatronModuleForCausalLM 输出头，并定义了新的 generate 方法。需要注意的是，GPTModel 的输出是模型根据输入预测的下一个 token 的概率分布，维度为 [batch_size, seq_len, vocab_size]，然后通过采样方法确定最终输出的 token。

## 核心生成逻辑

进一步查看 self.infer_model.generate() 方法的实现，即 MegatronModuleForCausalLM.generate 方法：

```python
class MegatronModuleForCausalLMABC(torch.nn.Module, abc.ABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """
    def __init__(self,) -> None:
        ...

    def generate(self, input_ids=None, **kwargs):
        ...
        # tokenize the prompts
        context_tokens_tensor, context_length_tensor = self.tokenize_prompts(tokenizer=self.tokenizer,
                                                                             prompts=input_ids,
                                                                             tokens_to_generate=self.max_new_tokens,
                                                                             max_generate_length=self.max_length,
                                                                             add_BOS=False,
                                                                             broadcast=broadcast)

        # =======================================
        # Get the streaming tokens generator
        # =======================================
        if self.num_beams > 1:
            token_stream = self.beam_search_or_sampling(
                args.model[0],
                tokens=context_tokens_tensor,
                lengths=context_length_tensor,
                beam_size=self.num_beams,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                num_return_gen=self.num_return_sequences
            )
        else:
            token_stream = self.greedy_search_or_sampling(
                args.model[0],
                tokens=context_tokens_tensor,
                lengths=context_length_tensor,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                return_output_log_probs=self.return_output_log_probs
            )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        return self._token_generator(token_stream)
```

该方法首先对输入 prompt 进行 tokenize，然后根据参数选择不同的采样方法生成 token 流，最后通过 _token_generator 方法将 token 流转换为最终输出结果。核心的 token 生成逻辑都在 self.beam_search_or_sampling 和 self.greedy_search_or_sampling 方法中实现。

以 greedy search 为例，其核心生成循环如下：

```python
def generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        do_sample=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True):
    """Main token generation function.

    Args:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
    Note: Outside of model, other parameters only need to be available on
          rank 0.

    Returns: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        lengths: original prompt length, size: [b]
        output_log_probs: log probability of the tokens. size: [b, s, vocab_size]
    """

    # 获取全局参数和分词器。Megatron 使用全局变量来存储配置和组件，便于在不同模块间共享。
    args = get_args()
    tokenizer = get_tokenizer()

    # 获取批次大小、最小提示长度和最大序列长度。这些用于控制生成过程的循环范围
    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size) + " is greater than " + str(args.max_tokens_to_oom))

    # 创建推理上下文和前向步骤对象。这是 Megatron 中用于管理推理过程的组件。
    # forward step.
    inference_context = StaticInferenceContext(batch_size, max_sequence_length)
    forward_step = ForwardStep(model, inference_context)

    # 确定终止 token ID。eos_id 是 end-of-sequence ID，eod 是 end-of-document。
    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # 预分配内存。只在流水线的最后一个阶段分配，因为只有那里才有完整的 logits。
    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1, args.padded_vocab_size)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length

    # 跟踪每个序列是否已完成生成。
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============

    with torch.no_grad():
        # 构建注意力掩码和位置 ID。对于特殊任务 'needlebench' 使用不同的处理方式
        if getattr(args, "task", False) and args.task[0] == 'needlebench':
            micro_batch_size, seq_length = tokens.size()
            attention_mask = None
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
        else:
            attention_mask, position_ids = _build_attention_mask_and_position_ids(
                tokens)
        # 针对特定模型（如 Hunyuan）使用特殊的 pad ID 处理。
        if get_args().spec is not None and get_args().spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
            pad_id = 127961
            attention_mask = tokens.ne(pad_id)
        
        # 主生成循环，从最小提示长度开始，逐个生成 token。
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):
            # start of megatron_adaptation, here we change sample stratrgy
            # Pick the slice that we need to pass through the network.
            # KV Cache 优化：只处理新生成的 token，而不是整个序列（非KV cache方法），提高效率。
            if args.use_kv_cache:
                tokens2use = tokens[:, prev_context_length:context_length]
                positions2use = position_ids[:, prev_context_length:context_length]
                if attention_mask is not None:
                    attention_mask2use = attention_mask[
                        ..., prev_context_length:context_length, :context_length]
                else:
                    attention_mask2use = None
            else:
                tokens2use = tokens
                positions2use = position_ids
                attention_mask2use = attention_mask

            # 执行模型前向计算，得到 logits。
            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            # 只在流水线最后一个阶段处理 logits，因为只有那里才有完整的输出。
            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                if logits is None:
                    raise ValueError("logits must not be None for pipeline last stage")

                # Sample.
                # 如果采用了kv_cache，logits输出维度为[batch_size, 1, vocab_size]，否则logits的维度为[batch_size, seq_len, vocab_size]
                # 可以看出KV cache的计算复杂度为O(N),而常规的计算复杂度为O(N^2)
                if args.use_kv_cache:
                    last_token_logits = logits[:, -1, :]
                else:
                    last_token_logits = logits[:, context_length - 1, :]

                # 根据采样策略选择下一个 token。
                _, new_sample = _sample_strategy(last_token_logits,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature)

                # end of megatron_adaptation

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs:
                    last_token_logits = F.log_softmax(last_token_logits, dim=1)
                    output_log_probs[:, context_length - 1, :] = last_token_logits

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = torch.zeros(1, dtype=torch.uint8, device=torch.cuda.current_device())
            if mpu.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                done_token = (new_sample == termination_id).byte() & \
                        started.byte()

                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
                if get_expert_model_parallel_world_size() > 1:
                    pipeline_world_size = mpu.get_pipeline_model_parallel_world_size()
                    world_size = torch.distributed.get_world_size()
                    last_stage_first_rank = int((pipeline_world_size - 1) * world_size / pipeline_world_size)
                    torch.distributed.broadcast(done, last_stage_first_rank, mpu.get_tensor_and_data_parallel_group())                  

            if output_log_probs is None and not (getattr(args, "task", False) and args.task[0] == 'needlebench'):
                output_log_probs = torch.empty(output_log_probs_size,
                                        dtype=torch.float32,
                                        device=torch.cuda.current_device())
            # 关键部分：yield 生成中间结果，使函数成为 generator。这允许调用者逐步获取生成结果。
            yield tokens[:, :(context_length + 1)], lengths, output_log_probs

            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_eod_token_for_early_termination and done:
                break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================
    # 将最后一个流水线生成结果广播到第一个流水线阶段，作为新的输入
    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length, args.padded_vocab_size)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    return tokens, lengths, output_log_probs
```

通过这种设计，推理过程能够高效地逐 token 生成，并支持多种采样策略和优化技术（如 KV Cache）来提升推理效率。
