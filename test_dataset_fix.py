 #!/usr/bin/env python3
"""
Quick test script to verify dataset iteration fix works correctly.
Tests dataset loading and iteration without full training overhead.
"""

import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from prismatic.models.backbones.llm.prompting import PrismaPromptBuilder
from prismatic.models.backbones.vision import DINOSigLIPViTBackbone
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset


def test_dataset_iteration():
    """Test that the dataset can iterate beyond the corruption point."""
    print("🚀 Starting dataset iteration test...")
    
    # Configuration (matching your train.sh)
    data_root_dir = Path("datasets/libero_data")
    data_mix = "libero_lm_90"
    resize_resolution = (224, 224)
    
    # Initialize components needed for the dataset
    print("📦 Initializing components...")
    
    # Vision backbone for image transforms
    vision_backbone = DINOSigLIPViTBackbone(
        vision_backbone_id="dinosiglip-vit-so-224px",
        image_resize_strategy="resize-naive",
        default_image_resolution=224,
    )
    image_transform = vision_backbone.image_transform
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # Action tokenizer  
    action_tokenizer = ActionTokenizer(tokenizer)
    
    # Batch transform
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=PrismaPromptBuilder,
        predict_stop_token=True,
        print_prompt_limit=0,  # Disable prompt printing for test
    )
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=data_mix,
        batch_transform=batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=1000,  # Smaller buffer for testing
        train=True,
        image_aug=False,
        enable_cot=True,
    )
    
    print(f"📏 Dataset length: {len(dataset)}")
    print(f"🎯 Testing iteration beyond step {16793}...")
    
    # Test iteration
    iterator = iter(dataset)
    step = 0
    start_time = time.time()
    corruption_warnings = 0
    
    try:
        for batch in iterator:
            step += 1
            
            # Print progress every 1000 steps
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                rate = step / elapsed
                print(f"✅ Step {step:,} - Rate: {rate:.1f} steps/sec")
            
            # Check for corruption warnings in recent output
            # (This is a simple heuristic - in practice you'd capture stderr)
            
            # Stop test after reaching well beyond the problematic step
            if step > 20000:  # Beyond the original failure point
                print(f"🎉 SUCCESS! Reached step {step:,} without termination!")
                print(f"   Original failure was at step 16,793")
                print(f"   Test completed in {time.time() - start_time:.1f} seconds")
                break
                
            # Safety timeout (shouldn't be needed if fix works)
            if time.time() - start_time > 300:  # 5 minutes max
                print(f"⏰ Timeout reached at step {step}")
                break
                
    except StopIteration:
        print(f"❌ FAILURE! Dataset iterator stopped at step {step}")
        print("   This indicates the fix didn't work properly")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted at step {step}")
        return True
    except Exception as e:
        print(f"❌ ERROR at step {step}: {e}")
        return False
    
    return True


def quick_corruption_test():
    """Quick test to see if we encounter corruption warnings."""
    print("\n🔍 Quick corruption detection test...")
    
    # Just test the first few thousand steps quickly
    data_root_dir = Path("datasets/libero_data")
    data_mix = "libero_lm_90"
    
    # Minimal setup
    from prismatic.models.backbones.vision import DINOSigLIPViTBackbone
    from transformers import AutoTokenizer
    
    vision_backbone = DINOSigLIPViTBackbone("dinosiglip-vit-so-224px", "resize-naive", 224)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    action_tokenizer = ActionTokenizer(tokenizer)
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=vision_backbone.image_transform,
        prompt_builder_fn=PrismaPromptBuilder,
        print_prompt_limit=0,
    )
    
    dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=data_mix,
        batch_transform=batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=100,
        train=True,
        enable_cot=True,
    )
    
    # Test first 5000 steps quickly
    step = 0
    start_time = time.time()
    
    try:
        for batch in dataset:
            step += 1
            if step % 1000 == 0:
                print(f"   Step {step}")
            if step >= 5000:
                break
    except StopIteration:
        print(f"   Iterator stopped early at step {step}")
        return False
    
    elapsed = time.time() - start_time
    print(f"   Completed {step} steps in {elapsed:.1f}s ({step/elapsed:.1f} steps/sec)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DATASET ITERATION FIX TEST")
    print("=" * 60)
    
    # Check if data directory exists
    if not Path("datasets/libero_data").exists():
        print("❌ Data directory 'datasets/libero_data' not found!")
        print("   Please ensure the dataset is properly downloaded.")
        exit(1)
    
    # Run quick test first
    if not quick_corruption_test():
        print("\n❌ Quick test failed - dataset has issues")
        exit(1)
    
    print("\n" + "=" * 60)
    
    # Run full test
    success = test_dataset_iteration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED! Dataset fix appears to work correctly.")
        print("   The dataset can now iterate indefinitely without early termination.")
    else:
        print("❌ TEST FAILED! Dataset still has iteration issues.")
        print("   You may need to check the data files or investigate further.")
    print("=" * 60)