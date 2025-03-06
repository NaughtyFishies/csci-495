[Intro](https://www.youtube.com/watch?v=wjZofJX0v4M) \
[Training](https://www.youtube.com/watch?v=9-Jl0dxWQs8)
# Transformers
* Transformers produce output by being a predictive text generator, appending their own predictions to queries then predicting again to produce output.
* After layers of perceptrons and attention, you have the last token from the query and its embedding make a prediction for the following word.
### System Prompt
* An initial piece of text to establish a setting for the model.
* "What follows is a coversation between a user and a helpful, very knowledgable AI assistant."
* Then the user gives a query: "Give me some ideas for what to do when visiting Santiago."
* Then the model responds by predicting what this "helpful AI" **WOULD** say.
### Unembedding
* We have an unembedding matrix which has every possible word in the embedding (for GPT3 about 50K)
* Multiply the last embedding vector in sequence with this matrix which will be used as a probability distribution to generate the next word.
### We have to Softmax to get a probabilty distribution
* Softmax $e^{x_{i}}/\sum_{n-0}^{N-1}e^{x_n}$
* Sometimes we add some Tempetature T where we can control how much the larger number dominate the distribution or if we want smaller numbers to be more significant.
* Inputs are called "logits"
