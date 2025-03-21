# Understanding Transformer

## To Start With
 The Transformer has revolutionized Natural Language Processing, ushering in dramatic changes not only in artificial intelligence but also in search engines, essay writing, and beyond. I've taken full advantage of this breakthrough—building GPT-2 from scratch and even attempting to pretrain BERT. Yet, a thought struck me: I know how to use these powerful tools, but I haven’t truly grasped the essence of artificial intelligence. That’s why I decided to visualize the core computations of the Transformer. <br><br>

 *NOTE : In this article, I am going to use the example sentence. ***Aerodynamics are for people who can't build engines***.<br> Also, tokenizer is not accurate. This is just for example.

## Single head attention
![attention equation.png](Images%2Fattention%20equation.png) <br>
According to the paper `Attention is all you need`, the transformer use the Attention mechanism to calculate the relation between each tokens.
As you can see in the equation, Q,K and V is calculated and returns Attention Score. `Q` stands for Query, calculated by multiplying Input_Embedding `X` and Weight `W_q`. `V` stands for Value, also calculated by multiplying Input_Embedding `X` and Weight `W_v`. Finally, `K` stands for Key, calculated by multiplying Input_Embedding `X` and Weight `W_k`.<br>
Everybody who are interested in Artificial Intelligence are familiar with the equation. But, do I really understand what's happening under the hood? <br>

![single_attention.png](Images%2Fsingle_attention.png) <br>
This is the image of Query and Key calculation, showing how strong the relation is with the example of ***Aerodynamics are for people who can't build***. My goal is to estimate the next word, which is ***engine***. Higher attention score is visualized with yellow, and lower with dark blue.
As you can see, token `people` and token `are` have a higher attention score. Which is plausible, because the verb `are` is defined by noun `people`.


