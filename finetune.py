import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

def main():
    model_id = "meta-llama/Llama-2-7b-hf" 

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Preparando para treinamento kbit
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit", 
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1,
        lr_scheduler_type="cosine", 
        warmup_ratio=0.03, 
        fp16=True,
        group_by_length=True,
    )
    
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="instruction", # Ajuste conforme formato do JSONL
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    model.save_pretrained("./adapter_model")
    print("Modelo adaptador salvo com sucesso em ./adapter_model")

if __name__ == "__main__":
    main()
