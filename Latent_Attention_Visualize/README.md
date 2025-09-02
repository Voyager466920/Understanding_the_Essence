# Understanding Latent Attention

## To Start With
 After the appearance of Transformer, bunch of 

 Note: In this article, I use the following example sentence: ***"Aerodynamics are for people who can't build engines."*** Also, note that the tokenizer is simplified for illustration purposes.

## Single head attention
![attention equation.png](Images%2Fattention%20equation.png)<br>
 According to the paper Attention is All You Need, the core idea of the Transformer mechanism is the attention mechanism. Essentially, attention calculates the relationships between tokens. As shown in the equation, the attention score is computed using Q, K, and V.
- Query (Q): Obtained by multiplying the input embedding X with the weight W_q.
- Key (K): Computed by multiplying X with the weight W_k.
- Value (V): Computed by multiplying X with the weight W_v. <br>
While most who are interested in AI are familiar with this equation, I wondered: do I really understand what’s happening under the hood?

![single_attention.png](Images%2Fsingle_attention.png) <br>
In this image, the relationships between Query and Key are visualized for the example sentence "Aerodynamics are for people who can't build", with the goal of predicting the next word, "engine". Higher attention scores are highlighted in yellow, and lower scores in dark blue. Notice that the tokens "people" and "are" have higher attention scores—a plausible outcome since the verb "are" is closely related to the noun "people".
Here, Query, Key, and Value are derived from the embedding matrices. In GPT-2, each token is represented by a 768-dimensional vector.

### Calculating Query, Key and Value.
![InputX.png](Images%2FInputX.png) <br>
![QKVequation.png](Images%2FQKVequation.png) <br>
The above image visualizes the input embedding X, where each token is transformed into a 768-dimensional vector. 
This vector is then matrix-multiplied with the respective weights for Query, Key, and Value. These weights are learned during training. 
Typically, the weights have a shape of (768, 64), resulting in Q, K, and V matrices of size (9, 64) (assuming 9 tokens in the sentence).
The following illustrations show this process: <br>
![Query.png](Images%2FQuery.png)
![Key.png](Images%2FKey.png)
![Value.png](Images%2FValue.png) <br>
The relation between Query and Key is computed by the matrix multiplication of Q and the transpose of K, yielding a (9 × 9) matrix. 
The scaling factor, the square root of d_k (where d_k = embedding_dim / num_heads), normalizes the result of (QK^T). 
After applying softmax (with potential masking to prevent "cheating"), this probability matrix is multiplied by the Value matrix, resulting in an output of shape (9, 64) that helps the model capture context. <br>

### Expanding to Multihead Attention
In GPT-2, the model employs 12 heads and 12 layers of attention blocks. This process is visualized below: <br>
![12_head_gather.png](Images%2F12_head_gather.png)
<br>
Each attention head produces an output of shape (9, 64), and after concatenating all heads, the result is a (9, 768) matrix. 
This matrix is then multiplied by W_o, a weight matrix with the shape (embedding_dim, embedding_dim). 
The final attention score has a shape of (9, 768) and is further processed through layer normalization, residual connections, and an MLP. Since the Transformer is trained end-to-end, all weights (W_q, W_k, W_v, W_o, and the MLP weights) are learned during training.

![GPT_heatmap.png](Images%2FGPT_heatmap.png) <br>
The image above shows an actual GPT-2 heatmap for Query and Key across 12 heads and 12 layers. 
Each head’s output is normalized with softmax, multiplied by Value, and then stacked through multiple layers before being passed to the MLP layer.<br>

This is a visualization of how the Transformer operates at its core.
