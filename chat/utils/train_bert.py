from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import json

# Cargar dataset
with open("../data/training_dataset.json") as f:
    examples = [json.loads(line) for line in f]

# Convertir a formato Hugging Face Dataset
dataset = Dataset.from_dict({
    "context": [ex["context"] for ex in examples],
    "question": [ex["question"] for ex in examples],
    "answers": [ex["answers"] for ex in examples]
})

# Tokenizar con offsets
tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", use_fast=True)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Mapear posiciones de las respuestas a los tokens
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = examples["answers"][sample_idx]

        # Encontrar los tokens que contienen la respuesta
        start_char = answer["answer_start"][0]
        end_char = answer["answer_end"][0]

        token_start_idx = 0
        while offsets[token_start_idx][0] <= start_char:
            token_start_idx += 1
        token_start_idx -= 1

        token_end_idx = len(offsets) - 1
        while offsets[token_end_idx][1] >= end_char:
            token_end_idx -= 1
        token_end_idx += 1

        tokenized["start_positions"].append(token_start_idx)
        tokenized["end_positions"].append(token_end_idx)

    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Cargar modelo
model = BertForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="models/finetuned_bert",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="logs",
    save_strategy="epoch",
    learning_rate=3e-5,
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Entrenar
trainer.train()

model.save_pretrained("models/finetuned_bert")
tokenizer.save_pretrained("models/finetuned_bert")