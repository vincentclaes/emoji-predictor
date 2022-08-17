#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# +

# # Fine Tune CLIP on Tweets
# 
# - CLIP on huggingface: https://huggingface.co/openai/clip-vit-base-patch32
# - Dataset: https://huggingface.co/datasets/AlekseyDorkin/extended_tweet_emojis/tree/main
# 
# ## 1. Install Dependencies

# +


# you might want to restart the kernel
# coupling between torch and torchvision: https://pypi.org/project/torchvision/
# get_ipython().system('pip install torchvision==0.11.1 torch==1.10.0 --quiet')


# +


# you might want to restart the kernel after installation is complete.
# get_ipython().system('pip install transformers datasets pillow ipywidgets requests jupyter jupyter_client wandb sklearn --upgrade --quiet')


# ## 2. Init Variables and Tools

# +
from IPython import get_ipython

# pwd = get_ipython().getoutput('pwd')
data_path = "./emojis"
data_path


# +


# import wandb
from transformers import TrainingArguments, Trainer
import os
os.environ["WANDB_DISABLED"] = "true"
# wandb.init(project="emoji-predictor", entity="drift-ai")


# ## 3. Setup Data Preprocessors and Trainer

# +


from pathlib import Path

import torch
from torchvision.transforms import Resize, InterpolationMode, ConvertImageDtype, CenterCrop
from torchvision.io import read_image, ImageReadMode
from transformers import CLIPProcessor, CLIPModel, Trainer
from transformers import default_data_collator, TrainingArguments
from datasets import load_dataset, Dataset

# Loading Data
dataset = load_dataset("vincentclaes/emoji-predictor")
train_dataset = dataset["train"].select(range(16))
val_dataset = dataset["validation"].select(range(16))
test_dataset = dataset["test"]

column_names = train_dataset.column_names
assert "label" in column_names
assert "text" in column_names
image_column = "label"
caption_column = "text"


# +


# Loading Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
config = model.config
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor

MAX_TEXT_LENGTH=77
IMAGE_SIZE = config.vision_config.image_size

# Preprocess Text
def tokenize_captions(examples):
    captions = [caption for caption in examples[caption_column]]
    text_inputs = tokenizer(captions, max_length=MAX_TEXT_LENGTH, padding="max_length", truncation=True)
    examples["input_ids"] = text_inputs.input_ids
    examples["attention_mask"] = text_inputs.attention_mask
    return examples


train_dataset = train_dataset.map(
    function=tokenize_captions,
    batched=True,
    remove_columns=[col for col in column_names if col != image_column],
    num_proc=None,
    load_from_cache_file=False,
    desc="Running tokenizer on train dataset",
)

val_dataset = val_dataset.map(
    function=tokenize_captions,
    batched=True,
    remove_columns=[col for col in column_names if col != image_column],
    num_proc=None,
    load_from_cache_file=False,
    desc="Running tokenizer on val dataset",
)

test_dataset = test_dataset.map(
    function=tokenize_captions,
    batched=True,
    remove_columns=[col for col in column_names if col != image_column],
    num_proc=None,
    load_from_cache_file=False,
    desc="Running tokenizer on test dataset",
)
#
# # Preprocess Images
# class Transform(torch.nn.Module):
#     def __init__(self, image_size):
#         super().__init__()
#         self.transforms = torch.nn.Sequential(
#             # resize and then crop the image to the image_size
#             Resize([image_size], interpolation=InterpolationMode.BICUBIC),
#             CenterCrop(image_size),
#             # convert RGB to floats
#             ConvertImageDtype(torch.float),
#         )
#
#     def forward(self, x) -> torch.Tensor:
#         with torch.no_grad():
#             x = self.transforms(x)
#         return x
#
#
# image_transformations = Transform(
#     IMAGE_SIZE
# )
# image_transformations = torch.jit.script(image_transformations)


from PIL import Image

def transform_images(examples):
    # https://pytorch.org/vision/stable/_modules/torchvision/io/image.html#ImageReadMode
    images = [Image.open(str(Path(data_path,f"{c}.png"))) for c in examples[image_column]]
    images_transformed = processor.feature_extractor(images, return_tensors="pt")

    # images = [read_image(str(Path(data_path,f"{c}.png")), ImageReadMode.RGB) for c in examples[image_column]]
    examples["pixel_values"] = images_transformed["pixel_values"]
    return examples


train_dataset.set_transform(transform_images)
val_dataset.set_transform(transform_images)
test_dataset.set_transform(transform_images)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    # labels = torch.tensor([example["label"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # "labels": labels,
        "return_loss": True,
    }

# def compute_metrics(p):
#     is_regression = False
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
#     if data_args.task_name is not None:
#         result = metric.compute(predictions=preds, references=p.label_ids)
#         if len(result) > 1:
#             result["combined_score"] = np.mean(list(result.values())).item()
#         return result
#     elif is_regression:
#         return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
#     else:
#         return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

from datasets import load_metric
import numpy as np
metric = load_metric("precision")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./checkpoints",
                           weight_decay=0.1,
                           dataloader_num_workers=0,
                           per_device_eval_batch_size=8,
                           per_device_train_batch_size=8,
                           num_train_epochs=1,
                           warmup_steps=0,
                           learning_rate=5e-05,
                           # report_to="wandb",
                           # metric_for_best_model="accuracy",
                           # label_smoothing_factor=0.0
                           ),
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    tokenizer=processor
)


# +


for batch in trainer.get_eval_dataloader(val_dataset):
    batch


# +


print(batch)


# ## 4. Train the Model

# +


from transformers.trainer_utils import get_last_checkpoint
# train_result = trainer.train(resume_from_checkpoint=get_last_checkpoint("./checkpoints"))
train_result = trainer.train()


# +


train_result.metrics


# +


trainer.evaluate(ignore_keys=["text_model_output", "vision_model_output", "text_embeds", "logits_per_image"])


# +


trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


# +


# metrics = trainer.evaluate()
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)


# +

#
# kwargs = {
#     "finetuned_from": "emoji-predictor",
#     "tasks": "contrastive-image-text-modeling",
#     "dataset": "AlekseyDorkin/extended_tweet_emojis"
# }
# trainer.push_to_hub(**kwargs)
trainer
