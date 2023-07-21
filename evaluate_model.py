import torch
from transformers import BertTokenizer, BertForQuestionAnswering,AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import pipeline
from datasets import load_dataset
import collections

def generate_response(question,context, max_length=50):
    input_ids = tokenizer.encode(question, context,return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def calculate_em_f1_score(model, tokenizer, dataset):
    """
    Calculate Exact Match (EM) score and F1 score for a Question Answering model on the SQuAD dataset.
    
    Args:
        model (transformers.PreTrainedModel): The QA model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding inputs.
        dataset (datasets.Dataset): The SQuAD dataset.

    Returns:
        float: Exact Match (EM) score.
        float: F1 score.
    """
   

    total_em, total_f1 = 0, 0
    total_examples = len(dataset)

    for example in dataset:
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]  # For SQuAD, we consider only the first answer

        prediction = generate_response(question=question, context=context)
        
        predicted_answer = prediction
        print(predicted_answer)
        # Calculate EM and F1 score for the current example
        em_score = int(predicted_answer == answer)
        f1_score = squad_f1(predicted_answer, answer)

        total_em += em_score
        total_f1 += f1_score

    # Calculate average EM and F1 scores over all examples
    em_score_avg = total_em / total_examples
    f1_score_avg = total_f1 / total_examples

    return em_score_avg, f1_score_avg

def squad_f1(pred, target):
    """
    Calculate F1 score for a given predicted answer and target answer (ground truth).

    Args:
        pred (str): Predicted answer.
        target (str): Target answer (ground truth).

    Returns:
        float: F1 score.
    """
    common = collections.Counter(target.split()) & collections.Counter(pred.split())
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred.split())
    recall = 1.0 * num_same / len(target.split())
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

if __name__ == "__main__":
    # Load the model and tokenizer (you can use any QA model and tokenizer)
    # model_name = "bert-base-uncased"
    model_path = "./t5_qa/"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

    # Load the SQuAD dataset
    dataset = load_dataset("squad")

    # Calculate EM and F1 scores
    em_score, f1_score = calculate_em_f1_score(model, tokenizer, dataset["validation"])

    print("Exact Match (EM) score:", em_score)
    print("F1 score:", f1_score)