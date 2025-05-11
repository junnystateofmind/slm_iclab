import csv
import argparse
import os
import random
import string

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,  # Function to prepare for K-bit
)
import bitsandbytes.optim as bnb_optim

def main(args):
    # ---------------------------------------------------------------------------
    # Path settings
    root = args.root_path
    assert root != "", "Please provide the root path using --root_path"

    data_path = os.path.join(root, "Mental_Emotion_Dataset/chatgpt_4o_mini_instructed")
    assert os.path.exists(data_path), "The provided data path does not exist."

    # QLoRA settings
    use_qlora = True
    lora_r = 256
    lora_alpha = 512

    # Model settings
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    # Change to Qwen model
    token_path = "qwen/Qwen2.5-3B-Instruct"  # Change to Qwen model path
    model_dir = os.path.join(root, "pretrained_models", token_path.replace("/", "_"))

    # Check model path and download (including tokenizer)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading model to {model_dir}...")
        model_tmp = AutoModelForCausalLM.from_pretrained(
            token_path,
            trust_remote_code=True,  # Required when loading Qwen model
            # If additional arguments are needed, set them here
        )
        tokenizer_tmp = AutoTokenizer.from_pretrained(
            token_path,
            trust_remote_code=True  # Required when loading Qwen tokenizer
        )
        model_tmp.save_pretrained(model_dir)
        tokenizer_tmp.save_pretrained(model_dir)
        print("Model downloaded and saved.")
    else:
        print(f"Model already exists at {model_dir}.")

    # Training settings
    model_max_length = 512
    num_train_epochs = args.num_epochs

    # Output directory settings
    _tmp = token_path.replace("/", "-") + ("-QLoRA" if use_qlora else "")
    output_dir = os.path.join(root, "train_and_evaluate/Output", f"{random_id}-{_tmp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at {output_dir}")

    # --------------------------------------------------------------------------
    # Dataset definition
    class CausalLMDataset(Dataset):
        """
        This is a Dataset implemented as an example for Instruction Tuning,
        where the user Prompt part is excluded from Loss (treated as -100),
        and only the Answer (Assistant) part is trained.
        """

        def __init__(self, data_path, tokenizer, max_length):
            self.data = self.load_data(data_path)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def load_data(self, file_path):
            texts = []
            with open(file_path, "r", encoding="utf-8") as data_file:
                reader = csv.reader(data_file)
                for idx, row in enumerate(reader):
                    if idx == 0:  # Skip header
                        continue
                    if len(row) != 2:
                        continue
                    input_text, output_text = row
                    # Formatting for Prompt/Answer distinction
                    prompt = f"User: {input_text}\nAssistant:"
                    texts.append((prompt, output_text))
            print(f"Successfully loaded {len(texts)} examples from {file_path}")
            return texts

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            prompt, answer = self.data[idx]

            # Combine Prompt and Answer to create the full text
            combined_text = prompt + " " + answer

            # Tokenize, pad, and truncate the sequence using the tokenizer
            encoding = self.tokenizer(
                combined_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = encoding['input_ids'].squeeze(0)          # [max_length]
            attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]

            # Calculate the number of tokens in the Prompt part
            prompt_encoding = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_length = prompt_encoding['input_ids'].size(1)

            # Create labels: Mask the Prompt part with -100, only the Answer part has actual values
            labels = input_ids.clone()
            labels[:prompt_length] = -100

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    # BitsAndBytesConfig settings (4bit, NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        truncation_side='left',
        model_max_length=model_max_length,
        trust_remote_code=True  # Required when loading Qwen tokenizer
    )

    # If pad_token is not defined, use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset (pass tokenizer)
    train_dataset = CausalLMDataset(f"{data_path}/train.csv", tokenizer, model_max_length)
    test_dataset = CausalLMDataset(f"{data_path}/test.csv", tokenizer, model_max_length)

    # Model loading
    #   - If transformers>=4.31, quantization_config=bnb_config is recommended
    #   - Use device_map="auto" to attempt automatic distribution across multiple GPUs/CPUs
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,  # 4-bit quantization settings
        device_map="auto",               # When distributing across multiple GPUs/CPUs
        trust_remote_code=True           # Required when loading Qwen model
    )

    # Prepare the model for PEFT training (use K-bit preparation function)
    base_model = prepare_model_for_kbit_training(base_model)

    # Attempt to activate xFormers (or flash_attn) (only if installed)
    try:
        base_model.supports_xformers_memory_efficient_attention = True
        base_model.enable_xformers_memory_efficient_attention()
        print(">>> xFormers memory efficient attention is enabled.")
    except Exception as e:
        print(">>> xFormers not available or failed to enable. Proceeding without it.")

    print(f"Base model loaded with 4-bit quantization: {base_model.__class__.__name__}")

    # Apply LoRA (QLoRA)
    if use_qlora:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model

    print(f"Model prepared with QLoRA: {model.__class__.__name__}")

    # TrainingArguments
    # - Reduce memory with gradient_checkpointing=True
    # - Choose "epoch" or "steps" for evaluation_strategy/save_strategy as needed
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        seed=42,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,  # Memory saving

        learning_rate=args.learning_rate,
        bf16=True,

        eval_strategy="epoch",  # In transformers>=4.46, use `eval_strategy`
        save_strategy="epoch",  # In transformers>=4.46, use `save_strategy`
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,

        report_to="none",
    )

    # Use 8-bit optimizer (bnb_optim)
    optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, None),
    )

    # Proceed with training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=False, default="/home/iclab/minjun/Tiny-Mental-LLM", help="Root path for the project")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Model saving frequency")

    args = parser.parse_args()
    main(args)