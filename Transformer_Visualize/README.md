# Understanding Transformer

## To Start With
 The Transformer has revolutionized Natural Language Processing, ushering in dramatic changes not only in artificial intelligence but also in search engines, essay writing, and beyond. I've taken full advantage of this breakthrough—building GPT-2 from scratch and even attempting to pretrain BERT. Yet, a thought struck me: I know how to use these powerful tools, but I haven’t truly grasped the essence of artificial intelligence. That’s why I decided to visualize the core computations of the Transformer. <br><br>

 *NOTE : In this article, I am going to use the example sentence. ***Aerodynamics are for people who can't build engines***.<br> Also, tokenizer is not accurate. This is just for example.

## Single head attention
![attention equation.png](Images%2Fattention%20equation.png)<br>
According to the paper `Attention is all you need`, the transformer use the Attention mechanism to calculate the relation between each tokens.
As you can see in the equation, Q,K and V is calculated and returns Attention Score. `Q` stands for Query, calculated by multiplying Input_Embedding `X` and Weight `W_q`. `V` stands for Value, also calculated by multiplying Input_Embedding `X` and Weight `W_v`. Finally, `K` stands for Key, calculated by multiplying Input_Embedding `X` and Weight `W_k`.<br>
Everybody who are interested in Artificial Intelligence are familiar with the equation. But, do I really understand what's happening under the hood? <br>

![single_attention.png](Images%2Fsingle_attention.png) <br>
This is the image of Query and Key calculation, showing how strong the relation is with the example of ***Aerodynamics are for people who can't build***. My goal is to estimate the next word, which is ***engine***. Higher attention score is visualized with yellow, and lower with dark blue.
As you can see, token `people` and token `are` have a higher attention score. Which is plausible, because the verb `are` is defined by noun `people`.<br>
`Query` and `Key`, `Value` is calculated by embedded matricies. GPT2 has 768 embedding dimension which means *each token is modified into matrix with 768 numbers(dimensions).*

### Calculating Query, Key and Value.
![InputX.png](Images%2FInputX.png) <br>
![QKVequation.png](Images%2FQKVequation.png) <br>
  The upper image is visualizing the input embedding X. Each token modified into 768 dimensions. This value is matrix multiplied with Weight of each Query, Key and Value.
These weights are interanlly decided in the process of training. Weights have the shape of (768 * 64), making Q, K, V of (9 * 64) matricies. This is visualized like following:<br>
![Query.png](Images%2FQuery.png)
![Key.png](Images%2FKey.png)
![Value.png](Images%2FValue.png) <br>
 So the image is showing relation between Query and Key is calculated by matrix multipling Q and K transpose, making (9 * 9) matrix.
The square root of d_k is for scaling QK^T, calculated by dividing embedding dimension with number of heads (d_k = embedding dim / num heads). 
So, (QK^T)/sqrt(d_k) is calculating the similarities between Q and K, showing the relationship. <br>
After calculating (QK^T)/sqrt(d_k), softmax turns into probability. Masking is selective, based on task, which is added to ensure model not to cheat for true value.
Then, the Value is multiplied. The Matrix multiplication of Softmax((QK^T)/sqrt(d_k)) and Value makes the model to understand context. As a result, it has a shpe of (9 * 64).<br>

### Expanding to Multihead Attention
In case of GPT2, model has 12 heads and 12 layers of Attention block. This looks like the following image. <br>
![12_head_gather.png](Images%2F12_head_gather.png)
<br>
The resulting value of each attention head is a matrix of (9 * 768), this value is matrix multiplied with Weight_o. Weight_o has a shape of (embedding_dim, embedding_dim).
Finally, Attention score is calculated as (9 * 768). This value is passed through layer normalization, residual connection to MLP. <br>
Since Transformer model is a end-to-end model, W_q, W_k, W_v, W_o and other MLP weights are trained.<br>

![GPT_heatmap.png](Images%2FGPT_heatmap.png)
