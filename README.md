## Glossary

This document is a quick-reference guide of terms, definitions, and concepts featured in the course.



*   **Artificial intelligence (AI):** A broad field encompassing the development of computer systems capable of performing tasks that typically require human intelligence, such as learning, problem-solving, and decision-making.

*   **Batching:** A training technique where the dataset is divided into smaller groups (batches), and the model's parameters are updated after processing each batch. This improves computational efficiency and can lead to more stable training.


*   **Corpus (text):** A large and structured set of texts used for training language models and other natural language processing tasks.
*   **Continuation:** The text a language model generates following a given input, known as a **prompt**. The model produces this output by sequentially predicting the most probable next word or token, building upon the provided context.


*   **Dataset:** A collection of data, often organized in a structured format, used for training, evaluating, and testing machine learning models.
*   **Deterministic:** A system where the output is uniquely determined by the input. In language models, always selecting the token with the highest probability during decoding would be a deterministic (or "greedy") approach.


*   **Encode and decode functions:** In the context of sequence-to-sequence models (like some transformer architectures), the **encode** function processes the input sequence into a fixed-length representation (context vector), and the **decode** function generates the output sequence based on this context vector.
*   **Epoch:** A complete pass through the entire training dataset during the training process. Multiple epochs are often required for the model to learn effectively.
*   **Evaluation:** The process of assessing the performance of a machine learning model using specific metrics.


*   **Generation:** The overarching process of creating new text from scratch or in response to a specific prompt. This is a broader term than **continuation**. While continuation is a specific method of generation, generation can also refer to creating a summary, translating text, or writing a creative story based on a single instruction.


*   **Hyperparameters:** Parameters of a machine learning model that are set before the training process begins. They control various aspects of the training, such as the learning rate, batch size, and the number of layers in a neural network.


*   **Language model (LM):** A system that learns to predict the next word in a sequence based on previous words or offers a way to assign probability to a sequence of words.
*   **Large language model (LLM):** A language model with a very large number of parameters (the variables the model learns during training), typically resulting in enhanced capabilities in understanding and generating human-like text.
*   **Loss:** A measure of the error between the model's predictions and the actual target values during training. The goal of training is to minimize this loss.


*   **Machine learning:** A subfield of artificial intelligence that enables computer systems to learn from data without being explicitly programmed.


*   **N-grams:** A contiguous sequence of $n$ words in a text. Examples include:
    *   **Unigram:** One word (e.g., "the").
    *   **Bigram:** Two words (e.g., "the cat").
    *   **Trigram:** Three words (e.g., "the cat sat").
*   **Natural language processing (NLP):** A field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language.


*   **Optimization:** The process of adjusting the parameters of a machine learning model during training to minimize the loss function and improve its performance.


*   **Padding:** A technique used to ensure that all sequences within a batch have the same length. Shorter sequences are padded with special tokens (e.g., `<pad>`) so they can be processed uniformly.
*   **Parameters:** The internal variables of a machine learning model that are learned from the training data. These weights and biases are adjusted during the optimization process.
*   **Probabilities:** Numerical values between 0 and 1 representing the likelihood of an event occurring.
*   **Probability distribution:** A mathematical function that describes the likelihood of different outcomes. In language models, it represents the probability of different tokens occurring in a given context.
*   **Prompt:** An initial input given to a language model to elicit a desired response or generation of text.


*   **Small language model (SLM):** A language model with a relatively smaller number of parameters compared to large language models.
*   **Sparsity:** A characteristic of language data where most possible combinations of words do not occur, resulting in many zero or near-zero values in statistical representations.
*   **Stochastic:** A system that involves randomness, meaning the output is not fixed for a given input. In language models, stochastic decoding involves sampling from the probability distribution to introduce variability.


*   **Tokenization:** The process of breaking down text into smaller units called tokens, which can be words, subwords, or characters.
*   **Transformer models:** A type of neural network architecture that works auto-regressively by generating one token at a time and using the previous context for the next one.
*   **Truncation:** A technique used to shorten sequences that exceed a predefined maximum length.
*   **Tuples:** An ordered sequence of elements. In machine learning, data points are often represented as tuples of features.


*   **Unknown tokens:** Tokens (e.g., words) that were not encountered in the training data, often represented by a special `<unk>` token.
