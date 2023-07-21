from datasets import load_dataset,concatenate_datasets
from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftModelForSeq2SeqLM,LoraModel,LoraConfig
from huggingface_hub import HfFolder
from random import randrange
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import evaluate
import torch 
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
# Dataset 
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # rougeLSum expects newline after each sentence
    
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    
    result["gen_len"] = np.mean(prediction_lens)
    

    return result

def preprocess_function_test(examples, padding = "max_length"):
    questions = [q.strip() for q in examples["question"]]
    ans = [a['text'][0] for a in examples["answers"]]
    # print(ans)
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    # print(inputs[0])
    labels = tokenizer(text_target = ans , max_length=512, padding= "max_length", truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    inputs["labels"] = labels["input_ids"]
    return inputs

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
          
        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

if __name__ == "__main__":
    squad = load_dataset("squad")
    model_id = "google/flan-t5-small"
    dataset_id = "DemoQA"
    # print(squad)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_squad = squad.map(preprocess_function_test,batched=True,remove_columns=['id', 'title', 'context', 'question', 'answers'])
    print(tokenized_squad['train'][0])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,cache_dir="./",load_in_8bit=True,device_map="auto")
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
        )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
        )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Hugging Face repository id
    repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        # int8 = True, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=5,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        compute_metrics=compute_metrics,
    )
    training_output = trainer.train()
    model_path = 't5_qa_v2'
    loss_values = training_output.metrics["train_loss"]
    
    print("Loss at each epoch:", loss_values)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)