import os
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from datasets import Dataset, load_dataset
from qwen.configuration_qwen import QWenConfig
from qwen.modeling_qwen import QWenLMHeadModel

from qwen.tokenization_qwen import QWenTokenizer
# torch._dynamo.config.optimize_ddp = False
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

attn_implementation = "flash_attention_2"
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = "eager"



#1.训练数据来源

SFT_FILES = [
    "./datasets/aplca1.parquet",
    "./datasets/aplca2.parquet",
    "./datasets/aplca3.parquet",
]



#===================================================================================
# 以下为sft训练配置

@dataclass
class SFTConfig:
    max_seq_len: int = 512                                                                  #最大序列长度，默认为 512，表示模型输入序列的最大长度限制。
    sft_from_checkpoint_file: str = "./model_save/pre3/checkpoint-16600"#模型目录,表示model模型文件的存储路径。
    tokenizer_dir: str = "./model_save/pre3/checkpoint-16600"            #分词器目录,表示分词器模型文件的存储路径。                 
    
    model_save_dir: str = "./model_save/sft/"                                   #模型保存目录,表示微调模型文件的存储路径。
    
    #logs_dir: str = "./logs/"                                                                  #日志目录,表示日志文件的存储路径。   
    
    train_files: list = field(default_factory=lambda: TRAIN_FILES)         #训练文件列表，表示预训练数据的文件列表。
    eval_file: str = EVAL_FILE                                                              #评估文件，表示评估数据的文件路径。
    
    cache_dir = ".cache"
    # Windows 使用默认的attention实现，
    attn_implementation: str = ( "eager" if platform.system() == "Windows" else attn_implementation) 
    

#设置模板  

PROMPT_DICT = {
    "prompt_input": ("你是一个助手 " "用户: {instruction} {input} 回答: "),
    "prompt_no_input": ("你是一个助手 " "用户: {instruction}  回答: "),
}


def format_example(example):
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )
    if example.get("input"):
        target = example["output"] + "<|im_end|>"
        context = prompt_input.format_map(
            dict(instruction=example["instruction"], input=example["input"])
        )

        example["context"] = context
        example["target"] = target
    else:
        target = example["output"] + "<|im_end|>"
        context = prompt_no_input.format_map(dict(instruction=example["instruction"]))

        example["context"] = context
        example["target"] = target
    return example


#定义sft数据格式处理
def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    input_ids = tokenizer(prompt + target,return_tensors="pt", padding="longest",max_length=512, truncation=True,)
    seq_ids = tokenizer(prompt,return_tensors="pt", padding="longest",max_length=512, truncation=True,)
    input_ids_len = seq_ids.input_ids.ne(tokenizer.pad_token_id).sum().item()

    return {"input_ids": input_ids.input_ids[0], "seq_len": input_ids_len}



# 定义data_collator，mlm=False表示训练的是CLM模型
def data_collator(fetures):
    len_ids = [len(feture["input_ids"]) for feture in fetures]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    postion_ids_list = []
    labels_list = []
    for ids_l, feture in sorted(zip(len_ids, fetures), key=lambda x: -x[0]):
        ids = feture["input_ids"]
        seq_len = feture["seq_len"]
        labels = [-100] * seq_len + ids[seq_len:] + [-100] * (longest - ids_l)
        ids = ids + [tokenizer.im_end_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {"input_ids": input_ids, "labels": labels}

# 定义训练过程中的回调函数 N次log之后情况cuda缓存，能有效缓解低显存机器显存缓慢增长的问题
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()


def train_sft(config: pretrainConfig, peft_config: LoraConfig=None) -> None:
    """ 
    #step1. 加载训练好的tokenizer
    tokenizer = QWenTokenizer.from_pretrained(config.tokenizer_dir)
    #将 pad token 的标识符设置为 tokenizer.im_end_id
    tokenizer.pad_token_id = tokenizer.im_end_id  

    #词表大小设置64的整数倍
    vocab_size = len(tokenizer)
    if vocab_size % 64 != 0:
        vocab_size = (vocab_size // 64 + 1) * 64
    print(f"final vocab size: {vocab_size}")

    # 词表小于 65535用uint16存储，节省磁盘空间，否则用uint32存储
    map_dtype = np.uint16 if vocab_size < 65535 else np.uint32

    # token to id缓存到文件，使用时不用再次tokenize
    def token_to_id(samples: dict) -> dict:
        batch_txt = samples["text"]
        outputs = tokenizer(
            batch_txt,truncation=False,
            padding=False,return_attention_mask=False,
        )
        input_ids = [np.array(item[0:512], dtype=map_dtype) for item in outputs["input_ids"]]  #截断，防止padding太大
        return {"input_ids": input_ids}

    # print(token_to_id({'text':['判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。\n','下面是一篇文章的开头: "为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。']}))
   """
    #2.定义qwen 模型-读取config.json配置参数
    model = QWenLMHeadModel.from_pretrained(config.sft_from_checkpoint_file)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"QWen size: {model_size / 1000**2:.1f}M parameters")
 

    # 3. 加载SFT数据集
    tokenized_datasets = dataset.map(function=format_example, num_proc=32, keep_in_memory=False)
    my_datasets = tokenized_datasets.train_test_split(test_size=4096)
    train_dataset = my_datasets["train"]
    eval_dataset = my_datasets["test"]
    #4.定义data_collator，用于准备模型训练所需的输入数据，但不进行掩码语言建模任务。
    empty_cuda_cahce = EmptyCudaCacheCallback()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    #6. 定义训练参数

    print("#6. 定义训练参数########################################################")

    args = TrainingArguments(
        output_dir=args.model_save_dir,   #指定模型训练过程中输出文件的保存路径，包括日志、模型、评估结果等。
        per_device_train_batch_size=32,            #每个训练设备（GPU/TPU）的训练批量大小。

        gradient_accumulation_steps=2,            #梯度累积步数，用于将多个小批量的梯度累积后再进行参数更新。
        num_train_epochs=3,                        #训练的 epoch 数量，即遍历整个训练数据集的次数。
        weight_decay=0.1,                          #权重衰减（L2 正则化）的系数，用于控制模型参数的大小。
        ddp_find_unused_parameters=False,          #是否在分布式训练中查找未使用的参数。
        warmup_steps=0,                            #学习率预热步数，即在训练开始阶段逐渐增加学习率的步数。   
        learning_rate=6e-5,                        #初始学习率。
        evaluation_strategy="steps",                 #评估策略，可以是 "no"、"steps"或者 "epoch"。
        eval_steps=500,                              #评估步数，用于指定在 evaluation_strategy 为 "steps" 时每次评估的步数。
        save_steps=500,                               #保存模型的步数间隔，即每经过 save_steps 步训练后保存一次模型。
        save_strategy="steps",#保存策略，可以是 "steps"（按照 save_steps 指定的步数保存模型）或者 "epoch"（按照每个 epoch 保存一次模型）。#save_strategy="epoch",#
        save_total_limit=4,                          #保存模型的数量限制，即保存模型文件的最大数量。
        report_to="tensorboard",                  #指定报告输出的目标，例如 "tensorboard" 表示输出到 TensorBoard。
        optim="adamw_torch",                      #优化器的类型，例如 "adamw_torch" 表示使用 PyTorch 中的 AdamW 优化器。
        lr_scheduler_type="cosine",               #学习率调度器的类型，例如 "cosine" 表示使用余弦退火学习率调度器。
        bf16=True,                                #是否启用 bfloat16 混合精度训练。
        logging_steps=20,                            #记录日志的步数间隔。 
        log_level="info",                            #日志级别，例如 "info" 表示记录信息级别的日志。
        logging_first_step=True,                  #是否记录训练的第一步日志。
        # group_by_length=True,                   #是否根据样本长度对数据进行分组，可以在处理变长序列时提高训练效率。
        
   
    )
    
    # 7. 初始化 trainer
    trainer = Trainer(
        model=model,                              #训练的模型，即 QWenLMHeadModel。
        tokenizer=tokenizer,                      #分词器，用于将文本数据转换为模型输入的 tokens。  
        args=args,                                #训练参数，包括了模型训练的各种设置，如批量大小、学习率、优化器类型等。
        data_collator=data_collator,              #数据收集器，用于准备模型训练所需的输入数据。
        train_dataset=train_dataset,              #训练数据集，包含了用于训练模型的样本数据。
        eval_dataset=eval_dataset,                #评估数据集，包含了用于评估模型性能的样本数据。
        callbacks=[empty_cuda_cahce],          #回调函数列表，用于在训练过程中插入自定义的操作。
    )

    #8. 开始训练
    trainer.train()
    #继续从checkpoints训练代码
    #trainer.train( 'model_save/pre/checkpoint-3400', resume_from_checkpoint=True)

    # 9.计算困惑度Perplexity& save log
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    loss_log = pd.DataFrame(trainer.state.log_history)
    log_dir = './logs'
    loss_log.to_csv(f"{log_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # 10. 保存模型/lora
    trainer.save_model(args.model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))



if __name__ == "__main__":

    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model 
         inference_mode=False, 
         r=16, 
         lora_alpha=16, 
         lora_dropout=0.1, 
         bias="all",
    )

    pretrain_config = SFTConfig()

    train_sft(pretrain_config, peft_config=None)




