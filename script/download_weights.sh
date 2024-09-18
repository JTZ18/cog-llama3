#!/usr/bin/env python

import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    "jtz18/llama31-8b-peft-merged-jon", torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True,
    cache_dir=CACHE_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    "jtz18/llama31-8b-peft-merged-jon", padding_side="right", use_fast=False,
    cache_dir=CACHE_DIR
)