In order to generate using a model, navigate to the "gpt/models" directory by
```bash
cd gpt/models
```
Then from here we can run models by changing the model that is loaded within "run_model.py". To generate a story, we can type
```bash
python3 run_model.py
```
which will generate a single story. This is done by providing a "\<sos\>" token to start the story and generating until reaching an "\<eos\>" token.