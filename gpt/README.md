# Model Instructions
This model trains from "data/valid_strings.txt" and uses the vocabulary listed at "data/vocabulary". To train this model, run the following in the CLI in the gpt directory:
```bash
cd gpt
python3 train_model.py
```
By default, the model will save the trained model to the "gpt/models" directory. There is a block of commented out code within "gpt/train_model.py" that will immediatly use the trained model to generate 10,000 tokens and write it to a file named located at "gpt/generated.txt", so that you are able to see how the model is generating. 