import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil

import draccus
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"



@dataclass
class MergeConfig:
    # fmt: off
    base_dir: str = "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/"                            # Path to OpenVLA model (on HuggingFace Hub)
    adapter_dir: str = "/nvme1n1/liangzhi/openvla_checkpoint/maniskill-gas4-scFalse/_tmp_adapter/+mani_skill_rlds_dataset+b4+lr-2e-05+lora-r32+dropout-0.0/d1121_check/"
    save_dir: str = "./checkpoints/merge/"
    # fmt: on


@draccus.wrap()
def merge(cfg: MergeConfig) -> None:
    base_dir = Path(cfg.base_dir)
    adapter_dir = Path(cfg.adapter_dir)
    save_dir = Path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(base_dir), trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )

    print("Finish load pretrained model.")

    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)

    print("Finish load adapter model.")

    merged_vla = merged_vla.merge_and_unload()

    print("Finish merge adapter model.")

    # dataset_statistics.json
    shutil.copy(base_dir / "dataset_statistics.json", save_dir / "dataset_statistics.json")

    # Save processor and model weights to new directory
    processor.save_pretrained(save_dir)
    merged_vla.save_pretrained(save_dir)

    # process norm_states
    config = json.load((save_dir / "config.json").open())
    new_norm_stat = json.load((save_dir / "dataset_statistics.json").open())
    config["norm_stats"].update(new_norm_stat)
    json.dump(config, (save_dir / "config.json").open("w"), indent=2)



if __name__ == "__main__":
    merge()