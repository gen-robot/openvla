import os
import unittest
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import SFTConfig, SFTTrainer
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


class RLDSDatasetTester(unittest.TestCase):
    """Test the RLDS dataset integration with TRL SFTTrainer."""
    
    def setUp(self):
        # Create a minimal model and tokenizer for testing
        self.model_id = "openvla/openvla-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        # Path to mock RLDS data
        self.data_root_dir = Path("/home/gaofeng/arm_ws/EmbodiedAgent/quick_jump/openvla/datasets/libero_data") #Path(os.environ.get("TEST_RLDS_DATA_DIR", "datasets/rlds"))
        self.dataset_name = "libero_object_no_noops" #os.environ.get("TEST_RLDS_DATASET_NAME", "aloha_scoop_x_into_bowl")
        
        # Ensure the RLDS dataset is installed and available
        # For testing purposes, you might want to create a small mock dataset
        # that mimics the structure of your RLDS data
        
    def test_load_rlds_dataset_directly(self):
        """Test loading the RLDS dataset directly."""
        try:
            # Set up the batch transform
            batch_transform = RLDSBatchTransform(
                None,  # No action tokenizer needed for this test
                self.tokenizer,
                image_transform=lambda x: x,  # Identity transform for testing
                prompt_builder_fn=PurePromptBuilder,
                use_wrist_image=False,
                use_proprio=True,
                history_size=0
            )
            
            # Create the dataset
            dataset = RLDSDataset(
                data_root_dir=self.data_root_dir,
                data_mix=self.dataset_name, 
                batch_transform=batch_transform,
                resize_resolution=(224, 224),
                shuffle_buffer_size=100,
                train=True,
                image_aug=False,
            )
            
            # Check if the dataset is iterable
            batch = next(iter(dataset))
            self.assertIsNotNone(batch)
            self.assertIn("input_ids", batch)
            self.assertIn("images", batch)
            
            print(f"Successfully loaded RLDS dataset directly: {self.dataset_name}")
            
        except Exception as e:
            self.fail(f"Failed to load RLDS dataset directly: {e}")
    
    def test_load_rlds_dataset_with_load_dataset(self):
        """Test loading the RLDS dataset with load_dataset."""
        try:
            # Register your dataset path if needed
            # datasets.config.DATASETDICT_MODULES.update({"rlds_dataset": "path/to/rlds_dataset.py"})
            
            # Load the dataset
            dataset = load_dataset(
                "path/to/rlds_dataset.py",
                data_dir=str(self.data_root_dir),
                dataset_name=self.dataset_name,
                streaming=True,
                split="train"
            )
            
            # Check if the dataset is iterable
            batch = next(iter(dataset))
            self.assertIsNotNone(batch)
            self.assertIn("input_ids", batch)
            self.assertIn("images", batch)
            
            print(f"Successfully loaded RLDS dataset with load_dataset: {self.dataset_name}")
            
        except Exception as e:
            self.fail(f"Failed to load RLDS dataset with load_dataset: {e}")
    
    def test_rlds_dataset_with_sft_trainer(self):
        """Test using the RLDS dataset with SFTTrainer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Load the dataset with load_dataset
                dataset = load_dataset(
                    "path/to/rlds_dataset.py",
                    data_dir=str(self.data_root_dir),
                    dataset_name=self.dataset_name,
                    streaming=True,
                    split="train"
                )
                
                # Create a data collator
                def collate_fn(examples):
                    # Process vision inputs
                    input_ids = [example["input_ids"] for example in examples]
                    images = [example["images"] for example in examples]
                    
                    # Create padded batch
                    batch = self.tokenizer.pad(
                        {"input_ids": input_ids},
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Add image tensor
                    batch["pixel_values"] = torch.stack([torch.tensor(img.getdata()) for img in images])
                    
                    # Add actions if present
                    if "actions" in examples[0]:
                        batch["actions"] = torch.stack([torch.tensor(example["actions"]) for example in examples])
                    
                    # Labels are the same as input_ids for autoregressive training
                    batch["labels"] = batch["input_ids"].clone()
                    
                    return batch
                
                # Initialize SFT trainer
                training_args = SFTConfig(
                    output_dir=tmp_dir,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=1,
                    learning_rate=5e-5,
                    report_to="none",
                )
                
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=collate_fn,
                    processing_class=self.tokenizer,
                )
                
                # Run a single training step
                result = trainer.train()
                
                self.assertIsNotNone(result)
                print("Successfully tested RLDS dataset with SFTTrainer")
                
            except Exception as e:
                self.fail(f"Failed to use RLDS dataset with SFTTrainer: {e}")


if __name__ == "__main__":
    unittest.main()