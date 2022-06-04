# Emoji Predictor

Trying to predict emoji's based on a sentence, the unsupervised way.

## CLIP

Clip was really easy to work with thanks to jina.ai's clip as a service: https://github.com/jina-ai/clip-as-service 

1) You download the CLIP model
2) Embed the emoji's in the CLIP model
3) You send a sentence and CLIP predicts emoji's.

The results are not bad because we did not fine tune with any training data.
![image description](assets/clip.png)
