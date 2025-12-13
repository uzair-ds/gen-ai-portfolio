import json
import pandas as pd
import nltk
import evaluate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_data(filepath):
    with open(filepath, 'rb') as f:
        raw_data = json.load(f) 
    
    contexts = []
    questions = []
    answers = [] 
    num_q = 0
    num_pos = 0
    num_imp = 0 
        
    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                num_q  = num_q  +1
                if qa['is_impossible'] == True:
                    num_imp = num_imp +1
                else:
                    num_pos = num_pos +1
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    
    return contexts, questions, answers  


train_contexts, train_questions, train_answers = prepare_data('/home/ubuntu/uzair/beta_force/squad/train-v2.0.json')

train_answer_texts = [answer['text'] for answer in train_answers]
train_df  = pd.DataFrame({
    'question': train_questions,
    'answer': train_answer_texts,
    'context': train_contexts
})

# print("sample_question : ",train_questions[0])
# print("sample answer :", train_answers[0]['text'])
# print("context :",train_contexts[0])

valid_contexts, valid_questions, valid_answers = prepare_data("/home/ubuntu/uzair/beta_force/squad/dev-v2.0.json")

valid_answer_texts = [answer['text'] for answer in valid_answers]

test_df  = pd.DataFrame({
    'question': valid_questions,
    'answer': valid_answer_texts,
    'context': valid_contexts
})

train_df["source"] = train_df[["context", "question"]].apply(lambda x: "context-question-answering: context: " + x["context"] + " question: " + x["question"], axis=1)
train_df["target"] = train_df["answer"]
train_df = train_df[["source", "target"]]

test_df["source"] = test_df[["context", "question"]].apply(lambda x: "context-question-answering: context: " + x["context"] + " question: " + x["question"], axis=1)
test_df["target"] = test_df["answer"]
test_df = test_df[["source", "target"]]

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

from typing import List, Tuple
from collections import Counter
from tqdm import tqdm

model_id="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

from datasets import  load_metric
import numpy as np

nltk.download("punkt", quiet=True)
metric_rouge = evaluate.load("rouge")
metric_em = evaluate.load("exact_match")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result_rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result_em = metric_em.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"rouge": result_rouge, "exact_match": result_em}


def preprocess_function(sample,padding="max_length"):
    model_inputs = tokenizer(sample["source"], max_length=512, padding=padding, truncation=True,return_tensors="pt")
    labels = tokenizer(sample["target"], max_length=128, padding=padding, truncation=True,return_tensors="pt")
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
test_tokenized_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)
print(f"Keys of tokenized dataset: {list(train_tokenized_dataset.features)}")

lora_config = LoraConfig(
 r=8,
 lora_alpha=16,
 lora_dropout=0.1,
 bias="none",
 task_type="SEQ_2_SEQ_LM",
 target_modules=["q", "v"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

seed=12222
output_dir="./flant5-small-qa-0904"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    push_to_hub = False,
    predict_with_generate=True,
    fp16=False,
    report_to="tensorboard",
)

model.config.use_cache = False
import warnings
warnings.filterwarnings('ignore')

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

training_logs = trainer.state.log_history
with open("training_logs_qa.txt","w") as f:
    f.write(str(training_logs))

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

with open("eval_results.txt","w") as f:
        f.write(str(eval_results))

peft_save_model_id="flant5-small-qa_09042024"
trainer.model.save_pretrained(peft_save_model_id, push_to_hub=False)
tokenizer.save_pretrained(peft_save_model_id, push_to_hub=False)
trainer.model.base_model.save_pretrained(peft_save_model_id, push_to_hub=False)




