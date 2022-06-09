# CS224N default final project (2022 IID SQuAD track)

This work aims to investigate innovative designs of model architectures that can
help boost performance on the SQuAD 2.0 dataset, without using pre-trained
language models. Since around half of the questions are unanswerable, it is
important for the model to tactfully abstain from answering. The main contribution
is a self-implemented QANet architecture with extensions on the embedding layer
and the output layer. By using unified encoding for the context and question
before feeding into context-query attention and employing threshold-based answer
verification during testing, the model achieves stronger out-of-sample performance
than the original QANet baseline. With a novel debiased ensemble method, the
model achieves an EM score of 67.68 and F1 score of 70.53 on the test leaderboard
for the IID SQuAD track.

https://web.stanford.edu/class/cs224n/reports/default_116657437.pdf
