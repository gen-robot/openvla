#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenVLA RL Training Script
--------------------------
This script combines OpenVLA's existing SFT approach with trl library for RL training, 
focusing on RLDS data loading via datasets package.
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np
import tensorflow as tf
from PIL import Image

# OpenVLA imports
from prismatic.vla.materialize import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

# trl imports for RL training
from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig, PPOTrainer, PPOConfig
from trl.trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config
from trl.core import respond

# HF imports
from datasets import load_dataset, Dataset, IterableDataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoProcessor,
    DataCollatorWithPadding,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model

# Logging imports
import logging
from prismatic.overwatch import initialize_overwatch

# Configure logging
overwatch = initialize_overwatch(__name__)

# Configure TensorFlow to not use GPU 
tf.config.set_visible_devices([], "GPU")

@dataclass
class ModelConfig:
    """Model configuration for RL training"""
    model_name_or_path: str = field(
        default="", metadata={"help": "Path to pretrained OpenVLA model or checkpoint"}
    )
    vision_tower: Optional[str] = field(
        default=None, metadata={"help": "Vision tower model path (if different from the default in OpenVLA)"}
    )
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_rank: int = field(
        default=32, metadata={"help": "Rank of LoRA weight matrix"}
    )
    lora_alpha: int = field(
        default=64, metadata={"help": "Alpha parameter for LoRA"}
    )
    lora_dropout: float = field(
        default=0.0, metadata={"help": "Dropout applied to LoRA weights"}
    )
    use_quantization: bool = field(
        default=False, metadata={"help": "Whether to quantize the model"}
    )
    torch_dtype: str = field(
        default="auto", metadata={"help": "Torch dtype to use"}
    )
    freeze_vision_backbone: bool = field(
        default=True, metadata={"help": "Whether to freeze vision backbone"}
    )
    freeze_llm_backbone: bool = field(
        default=False, metadata={"help": "Whether to freeze language model backbone"}
    )
    unfreeze_last_llm_layer: bool = field(
        default=True, metadata={"help": "Whether to unfreeze last LLM layer"}
    )
    attn_implementation: Optional[str] = field(
        default=None, metadata={"help": "Attention implementation to use"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "Revision of the model to use"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "List of module names to apply LoRA to"}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Whether to trust remote code"}
    )

@dataclass
class DataConfig:
    """Data configuration for RL training"""
    data_root_dir: str = field(
        default="", metadata={"help": "Path to RLDS dataset directory"}
    )
    dataset_name: str = field(
        default="", metadata={"help": "Name of the dataset to use"}
    )
    data_mix: str = field(
        default="", metadata={"help": "Data mixture specification"}
    )
    shuffle_buffer_size: int = field(
        default=100000, metadata={"help": "Size of shuffle buffer"}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for training"}
    )
    image_aug: bool = field(
        default=True, metadata={"help": "Whether to use image augmentation"}
    )
    train_split: str = field(
        default="train", metadata={"help": "Split to use for training"}
    )
    eval_split: str = field(
        default="validation", metadata={"help": "Split to use for evaluation"}
    )

@dataclass
class TrainingConfig:
    """Training configuration for RL training"""
    output_dir: str = field(
        default="./outputs", metadata={"help": "Output directory"}
    )
    run_name: str = field(
        default="openvla_rl", metadata={"help": "Name of the run"}
    )
    training_mode: str = field(
        default="sft", metadata={"help": "Training mode: 'sft', 'grpo', or 'ppo'"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay"}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm"}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X steps"}
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Evaluate every X steps"}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Number of warmup steps"}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Whether to use fp16 precision"}
    )
    bf16: bool = field(
        default=True, metadata={"help": "Whether to use bf16 precision"}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed"}
    )
    report_to: str = field(
        default="wandb", metadata={"help": "Where to report results (none, wandb, tensorboard)"}
    )
    max_steps: int = field(
        default=-1, metadata={"help": "Maximum number of training steps. -1 means no limit."}
    )
    group_by_length: bool = field(
        default=False, metadata={"help": "Group sequences of similar length together to reduce padding."}
    )

class RLDSToDatasetAdapter(IterableDataset):
    """
    Adapter class that converts RLDS dataset to HF datasets format for use with trl
    """
    def __init__(
        self, 
        rlds_dataset, 
        action_tokenizer: ActionTokenizer, 
        tokenizer, 
        image_transform,
        prompt_builder_fn
    ):
        self.rlds_dataset = rlds_dataset
        self.action_tokenizer = action_tokenizer
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
    
    def __iter__(self):
        for batch in self.rlds_dataset:
            # Convert RLDS batch to format needed for trl
            # Extract image, language instruction, and action
            image = Image.fromarray(batch["observation"]["image_primary"][0])
            lang_instruction = batch["task"]["language_instruction"].decode().lower()
            action = batch["action"][0]
            
            # Process image with the image transform
            pixel_values = self.image_transform(image)
            
            # Convert action to text using action tokenizer
            action_text = self.action_tokenizer(action)
            
            # Create prompt using prompt builder
            prompt_builder = self.prompt_builder_fn("openvla")
            prompt_builder.add_turn("human", f"What action should the robot take to {lang_instruction}?")
            
            # Format for trl training
            yield {
                "pixel_values": pixel_values,
                "prompt": prompt_builder.get_prompt(),
                "completion": action_text,
                "lang_instruction": lang_instruction,
                "messages": [
                    {"role": "user", "content": f"What action should the robot take to {lang_instruction}?"},
                    {"role": "assistant", "content": action_text}
                ]
            }

class CustomCollator:
    """
    Custom data collator for OpenVLA model with RL training
    """
    def __init__(self, tokenizer, processor=None, max_length=512):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, examples):
        # Extract pixel values and text
        pixel_values = [example["pixel_values"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]
        
        # Tokenize
        prompt_ids = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Stack pixel values
        pixel_values = torch.stack(pixel_values)
        
        # Prepare completion tokens to be used as labels
        completion_ids = self.tokenizer(
            completions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Create labels by replacing prompt tokens with -100
        labels = completion_ids.clone()
        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
            
        return {
            "input_ids": prompt_ids.input_ids,
            "attention_mask": prompt_ids.attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }

def setup_reward_model(model_config):
    """
    Setup a reward model for RL training
    """
    # In a real implementation, load a trained reward model
    # Here, we define a simple reward function based on sequence length
    def simple_reward_function(outputs, **kwargs):
        # Example reward function - rewards shorter responses (as a proxy for efficiency)
        # In practice, you would use a trained reward model
        scores = []
        for output in outputs:
            # Simple length-based reward (shorter responses get higher reward)
            length = len(output.split())
            # Normalize to 0-1 range (assuming most responses are <100 tokens)
            normalized_score = max(0, min(1, 1 - (length / 100)))
            scores.append(normalized_score)
        return scores
    
    return simple_reward_function

def load_openvla_model(model_config):
    """
    Load the OpenVLA model with appropriate configuration
    """
    # This is a placeholder - in a real implementation, use OpenVLA's model loading logic
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=getattr(torch, model_config.torch_dtype) if model_config.torch_dtype != "auto" else "auto",
        device_map="auto"
    )
    
    # Apply LoRA if needed
    if model_config.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.target_modules or ["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
    
    # Freeze or unfreeze components based on config
    if model_config.freeze_vision_backbone:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    if model_config.freeze_llm_backbone:
        for name, param in model.named_parameters():
            if "language_model" in name and "lm_head" not in name:
                param.requires_grad = False
    
    if model_config.unfreeze_last_llm_layer:
        for name, param in model.named_parameters():
            if "language_model.model.layers.31" in name:  # Assumes 32 layers, adjust as needed
                param.requires_grad = True
    
    return model, processor

def setup_rlds_dataset(data_config, tokenizer, image_transform, prompt_builder_fn):
    """
    Set up RLDS dataset for training with trl
    """
    # Configure RLDS dataset from OpenVLA
    dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_root_dir=Path(data_config.data_root_dir),
        data_mix=data_config.data_mix,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        default_image_resolution=(3, 224, 224),  # Adjust as needed
        padding_side="right",
        predict_stop_token=True,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        train=True,
        episodic=False,
        image_aug=data_config.image_aug
    )
    
    # Adapt dataset to HF format
    hf_dataset = RLDSToDatasetAdapter(
        dataset, 
        action_tokenizer, 
        tokenizer, 
        image_transform,
        prompt_builder_fn
    )
    
    return hf_dataset, action_tokenizer, collator

def main():
    parser = argparse.ArgumentParser(description="OpenVLA RL Training")
    
    # Model configuration arguments
    parser.add_argument("--model_name_or_path", type=str, default="openvla/openvla-7b", 
                       help="Path to pretrained OpenVLA model or checkpoint")
    parser.add_argument("--vision_tower", type=str, default=None,
                       help="Vision tower model path (if different from the default in OpenVLA)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="Rank of LoRA weight matrix")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="Dropout applied to LoRA weights")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Whether to quantize the model")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       help="Torch dtype to use")
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=True,
                       help="Whether to freeze vision backbone")
    parser.add_argument("--freeze_llm_backbone", action="store_true", default=False,
                       help="Whether to freeze language model backbone")
    
    # Data configuration arguments
    parser.add_argument("--data_root_dir", type=str, default="",
                       help="Path to RLDS dataset directory")
    parser.add_argument("--dataset_name", type=str, default="",
                       help="Name of the dataset to use")
    parser.add_argument("--data_mix", type=str, default="",
                       help="Data mixture specification")
    parser.add_argument("--shuffle_buffer_size", type=int, default=100000,
                       help="Size of shuffle buffer")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--image_aug", action="store_true", default=True,
                       help="Whether to use image augmentation")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Split to use for training")
    parser.add_argument("--eval_split", type=str, default="validation",
                       help="Split to use for evaluation")
    
    # Training configuration arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--run_name", type=str, default="openvla_rl",
                       help="Name of the run")
    parser.add_argument("--training_mode", type=str, default="sft", choices=["sft", "grpo", "ppo"],
                       help="Training mode: 'sft', 'grpo', or 'ppo'")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every X steps")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="Whether to use fp16 precision")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Whether to use bf16 precision")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--report_to", type=str, default="wandb",
                       help="Where to report results (none, wandb, tensorboard)")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum number of training steps. -1 means no limit.")
    
    args = parser.parse_args()
    
    # Create configuration objects from args
    model_config = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        vision_tower=args.vision_tower,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_quantization=args.use_quantization,
        torch_dtype=args.torch_dtype,
        freeze_vision_backbone=args.freeze_vision_backbone,
        freeze_llm_backbone=args.freeze_llm_backbone
    )
    
    data_config = DataConfig(
        data_root_dir=args.data_root_dir,
        dataset_name=args.dataset_name,
        data_mix=args.data_mix,
        shuffle_buffer_size=args.shuffle_buffer_size,
        batch_size=args.batch_size,
        image_aug=args.image_aug,
        train_split=args.train_split,
        eval_split=args.eval_split
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        training_mode=args.training_mode,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
        max_steps=args.max_steps
    )
    
    # Set up the model
    model, processor = load_openvla_model(model_config)
    tokenizer = processor.tokenizer
    image_transform = processor.image_processor
    
    # Define prompt builder
    def create_prompt_builder(name):
        # This is a placeholder - in a real implementation, use OpenVLA's prompt builder
        class SimplePromptBuilder:
            def __init__(self, name):
                self.name = name
                self.turns = []
            
            def add_turn(self, role, text):
                self.turns.append((role, text))
            
            def get_prompt(self):
                result = ""
                for role, text in self.turns:
                    if role == "human":
                        result += f"USER: {text}\n"
                    else:
                        result += f"ASSISTANT: {text}\n"
                return result
        
        return SimplePromptBuilder(name)
    
    # Set up dataset
    train_dataset, action_tokenizer, _ = setup_rlds_dataset(
        data_config, 
        tokenizer, 
        image_transform, 
        create_prompt_builder
    )
    
    # Create custom collator
    data_collator = CustomCollator(tokenizer, processor)
    
    # Configure training
    if training_config.training_mode == "sft":
        # SFT configuration
        training_args = SFTConfig(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            max_grad_norm=training_config.max_grad_norm,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            warmup_steps=training_config.warmup_steps,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            seed=training_config.seed,
            report_to=training_config.report_to,
            run_name=training_config.run_name,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_steps=training_config.max_steps if training_config.max_steps > 0 else None,
            group_by_length=training_config.group_by_length,
        )
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
    elif training_config.training_mode == "grpo":
        # GRPO configuration
        training_args = GRPOConfig(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            max_grad_norm=training_config.max_grad_norm,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            warmup_steps=training_config.warmup_steps,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            seed=training_config.seed,
            report_to=training_config.report_to,
            run_name=training_config.run_name,
            remove_unused_columns=False,
            max_steps=training_config.max_steps if training_config.max_steps > 0 else None,
        )
        
        # Set up reward model
        reward_model = setup_reward_model(model_config)
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            reward_funcs=reward_model,
            data_collator=data_collator,
        )
    
    elif training_config.training_mode == "ppo":
        # PPO configuration
        training_args = PPOConfig(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            max_grad_norm=training_config.max_grad_norm,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            warmup_steps=training_config.warmup_steps,
            fp16=training_config.fp16,
            seed=training_config.seed,
            report_to=training_config.report_to,
            remove_unused_columns=False,
        )
        
        # Set up reward model
        reward_model = setup_reward_model(model_config)
        
        # Create PPO trainer
        trainer = PPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            reward_model=reward_model,
            data_collator=data_collator,
        )
    
    else:
        raise ValueError(f"Unsupported training mode: {training_config.training_mode}")
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(training_config.output_dir)
    
    print(f"Training complete! Model saved to {training_config.output_dir}")

if __name__ == "__main__":
    main()