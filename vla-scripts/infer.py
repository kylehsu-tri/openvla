import ipdb
import json
import pathlib
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from torch.utils.data import DataLoader

import prismatic
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

def infer():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    openvla_path = ("/home/ashwinbalakrishna/kylehsu/code/fix-droid/openvla/runs/openvla-7b+droid_pick_up_can_target"
                    "+b8+lr-2e-05+lora-r32+dropout-0.0+steps-45e3")
    processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        openvla_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    with open(pathlib.Path(openvla_path) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )

    dataset = RLDSDataset(
        "/home/ashwinbalakrishna/tensorflow_datasets",
        "droid_pick_up_can_target",
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        train=True,
        image_aug=False,

    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )


    batch = next(iter(dataloader))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"],
        )
        loss = output.loss
        ipdb.set_trace()


if __name__ == "__main__":
    infer()