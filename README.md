# Emoji Predictor

The idea is that we would like to predict emoji's based on a sentence.

The [__CLIP model (Contrastive Language-Image Pretraining)__](https://github.com/openai/CLIP) is a good candidate.

CLIP is a model of OpenAI trained on 400 M image and text pairs. 

So  what can it do?

1) Given an image, CLIP can be used to find a text snippet that describes the image.
2) Or the reverse, return an image given a sentence.

## Zero-shot 

Using the pretrained model using jina.ai's clip-as-a-service was super easy!

Go to [zero-shot](./zero-shot) to learn more.

## Fine Tune

Fine tuning CLIP on a few samples, but also on bigger data.

Go to [fine-tune](./fine-tune) to learn more.