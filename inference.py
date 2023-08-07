from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model

peft_model_id = "/ckpt/"
max_memory = {0: "80GIB", 1: "80GIB", "cpu": "30GB"}
config = PeftConfig.from_pretrained(peft_model_id)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", max_memory=max_memory, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto", max_memory=max_memory)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, legacy=False)
tokenizer.pad_token = tokenizer.unk_token
target_max_length = 128
tokenized = tokenizer("There was a hobbit named David.")
tokenized["input_ids"] = torch.tensor(tokenized["input_ids"]).unsqueeze(0)
tokenized["attention_mask"] = torch.ones(tokenized["input_ids"].size(1)).unsqueeze(0)
outputs = model.generate(input_ids=tokenized["input_ids"], max_new_tokens=1024, attention_mask=tokenized["attention_mask"])
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
with open('/output/out.txt', 'w') as f:
    f.write(result)
