import torch
from torchvision.transforms import Resize, InterpolationMode, ConvertImageDtype, CenterCrop
from transformers import CLIPProcessor, CLIPModel, Trainer
from transformers import default_data_collator, TrainingArguments
from datasets import load_dataset
from torchvision.io import read_image, ImageReadMode

# Loading Data
dataset = load_dataset("AlekseyDorkin/extended_tweet_emojis")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
column_names = train_dataset.column_names
assert "label" in column_names
assert "text" in column_names
image_column = "label"
caption_column = "text"

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


class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            # resize and then crop the image to the image_size
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            # convert RGB to floats
            ConvertImageDtype(torch.float),
        )

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


image_transformations = Transform(
    IMAGE_SIZE
)
image_transformations = torch.jit.script(image_transformations)


# Preprocess Images
def transform_images(examples):
    # https://pytorch.org/vision/stable/_modules/torchvision/io/image.html#ImageReadMode
    images = [read_image(f"../emojis/{c}.png", ImageReadMode.RGB) for c in examples[image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples


train_dataset.set_transform(transform_images)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./checkpoints",
                           weight_decay=0.1,
                           dataloader_num_workers=0,
                           per_device_eval_batch_size=64,
                           per_device_train_batch_size=64,
                           num_train_epochs=3.0,
                           warmup_steps=0,
                           learning_rate=5e-05,

                           ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn
)

trainer.train()
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)