from datasets import load_dataset,concatenate_datasets
from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,AutoModel
from peft import PeftModelForSeq2SeqLM,LoraModel,LoraConfig
from huggingface_hub import HfFolder
from random import randrange
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import evaluate
import numpy as np 

def generate_response(qs,context, max_length=50):
    input_ids = tokenizer.encode(qs, context,return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def generate_response_ori(qs,context, max_length=50):
    input_ids = tokenizer.encode(qs, context,return_tensors="pt")
    output = model_ori.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

model_path = "./t5_qa/"

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
model = AutoModelForSeq2SeqLM.from_pretrained(model_path,use_auth_token=True)

data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        max_length=512
    )

question = "What is the AFC short for?"
context = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""
out = generate_response(qs=question, context=context)
print(f"Fine tune model answer: {out}")
model_ori = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
out_ori = generate_response_ori(qs=question, context=context)
print(f"Original answer: {out_ori}")