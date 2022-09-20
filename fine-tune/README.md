# Emoji Predictor

Trying to predict emoji's based on a sentence.

- [00-preprocess-data.ipynb](./00-preprocess-data.ipynb): Preprocess the data we will use in the other notebooks.
- [01-zero-shot-with-CLIP-on-test-dataset-tweets.ipynb](./01-zero-shot-with-CLIP-on-test-dataset-tweets.ipynb): Apply the pretrained CLIP model to the test datatset and calculate the precision.
- [02-fine-tune-with-CLIP-on-train-dataset-tweets.ipynb](./02-fine-tune-with-CLIP-on-train-dataset-tweets.ipynb): Use the pretrained CLIP model and incrementally fine tune on bigger samples. We start with 1, 5, 10, ... up to 500 samples per emoji.
- [03-fine-tune-evaluate-with-CLIP-on-test-dataset-tweets.ipynb](./03-fine-tune-evaluate-with-CLIP-on-test-dataset-tweets.ipynb): Use the fine tuned models and apply them to the test dataset and calculate the precision.
- [04-play-with-fine-tuned-model.ipynb](./04-play-with-fine-tuned-model.ipynb): Take a fine tuned model and play around ...
