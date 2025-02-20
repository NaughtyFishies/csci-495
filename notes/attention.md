# Attention
[source](https://www.geeksforgeeks.org/ml-attention-mechanism/)
[video](https://www.youtube.com/watch?v=eMlx5fFNoYc)
## Attention Mechanisms
* Attention mechanism - A computer method for prioritizing specific information in a given context.
* Attention mechanisms allow neural networks to give various weights to various input items, boosting their ability to capture crucial information adn improve performance in a variety of tasks.
* An attention mechanism is an encoder-decoder kind of NN architecture that allows the model to focus on specific sections of the input while executing a task. It does this by dynamically assigning weights to different elements in the input, indicating their relative importance or relevance.
## Self Attention Head
* Start by encoding all of the tokens of the inputs into high dimensional vectors. These vectors allow each token to be unique but still have relationships to one another.
* The query for each token is the context related to that token. When each token attempts to determine its specific context about itself it creates a query vector which is a smaller vector than the embedding vector.
* To determine the query vector, we multiply the embedding vector by some matrix. This matrix is the same for all tokens. We take this matrix and multiply it by the embedding for each token to get the query vector for each token.
* Seperatly from this we multiply a key matrix with every embedding vector to get key vectors for all tokens.
* Think of the keys as matching the queries whenever they closely align with each other. We want the key matrix to map contextual tokens to the query of the token they are providing context to. EX: "The big, fluffy dog" - we want the key vectors of big and fluffy to align with the query vector of dog.
* To determine how well each key aligns with each query we compute a dot product of each key/query pair. Positive alignment correlate to a large positive number where negative alignment correlates to a negative number or a small positive number.
* We would say the embeddings of big/fluffy attend to the embedding of dog.
* We want these alignments to act like weights in our network, so we need to normalize each column so that they add to 1 like probabilities. 
* We can do this by applying softmax to each column.
* At this point we can think of each column as giving weights by how relevant some tokens are to others.
* Its helpful for training to allow your model to train on each next token as its recieving inputs, however we dont want later tokens to influence earlier predictions. So we essentially want the dot products of keys/queries to be 0 for tokens aligning with other tokens farther into the inputs. We can't set them all to 0 because then our columns/weights wouldnt be normalized. So we do this by before applying softmax we set these values to be $-\infin$, so then after the softmax they will all be 0.
#### Value Vectors
* At this point we want to update our embeddings. We want the embedding of dog to point somewhere that more specifically corresponds to a big/fluffy dog.
* To do this we create a value matrix, that we multiply by the embedding of the context tokens ie. big/fluffy to create a value vector. We then add this value vector to the embedding of dog to get its new embedding.
* In practice, we multiply each column of weights with each corresponding value vector. So tokens that correspond will get multiplied with larger numbers. We then sum each column to get the change in embedding, then add this change to the embedding of the token that this context applies to.
* In GPT3, the query and key vectors are 12,288 x 128 = 1,572,864 parameters big. The value matrix is formed as two matrices multiplied together each with the parameters of the key and query matrix repectively. Therefore each attention head of GPT3 had 12,288 x 128 x 4 = 6,291,456 parameters.
### Multiplicative Attention
* Multiplicative attention is faster than additive attention and more space efficient because it is implemented more efficiently using matrix multiplication.
* The process explained above using dot products is multiplicative attention.
* Additive attention uses a small feedforward neural network to compute attention scores.
* For large models using multiplicative attention is faster and more efficient.
* Additive attention is O(d) where multiplicative is O(1)
* Additive methods are used in RNNs where multiplicative methods are used in transformer models ie. self attention.
### Self Attention
* Self attention is used in transformers to map attention scores to itself which captures long-range dependencies faster.
* Self attention is also parallelizable so its more efficient.
* The example given above applies self attention, it attempts to find relationships inside the original query.