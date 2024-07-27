import numpy as np
import torch
import torch.nn.functional as F


def _mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take average of all tokens
    # see: https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1#pytorch-usage-huggingface-transformers
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def _tokenize(texts, tokenizer, max_length):
    """
    Generate token for input text.
    :param texts: a list of input sentences or texts to be processed
    :param tokenizer: a Hugging Face tokenizer instance
    :param max_length: an optional parameter for padding/truncation of text strings
    :return: encoded inputs
    """

    padding = True if max_length == 0 else 'max_length'

    # __call__ the tokenizer
    return tokenizer(
        texts,
        padding=padding,
        truncation=True,
        max_length=max_length,  # if left unset, uses model default
        return_tensors='pt'  # return as torch tensors
    )


def _embed(encoded_input, model):
    """
    Take tokenized values and generate embeddings
    :param encoded_input: encoded inputs generated by tokenizer
    :param model: a Hugging Face model instance
    :return: embedding vector
    """

    # Compute token embeddings
    with torch.no_grad():  # only need forward pass here
        embeddings_words = model(**encoded_input, return_dict=True)

    # Perform mean pooling
    # The attention mask ensures that padding tokens do not contribute to the averaged embedding.
    # purpose is to take variable length sequences and output fixed length ones
    embeddings_sentence = _mean_pooling(embeddings_words, encoded_input['attention_mask'])

    # Normalize embeddings -- L2 = 1
    embeddings_sentence = F.normalize(embeddings_sentence, p=2, dim=1)

    return embeddings_sentence, embeddings_words


def get_sent_word_embeddings(labels, model, tokenizer, max_length=None):
    """
    Generate sentence embedding for input texts
    :param model: Hugging Face model instance
    :param tokenizer: Hugging Face tokenizer instance
    :param labels: input labels from our training data
    :param max_length: maximum length for padding/truncation fo input stirngs
    :return: both mean-pooled sentence embedding and masked word embeddings
    """

    encoded = _tokenize(labels, tokenizer, max_length)
    embeddings_sentences, embeddings_words = _embed(encoded, model)

    embeddings_words = embeddings_words['last_hidden_state'].detach().cpu().numpy()
    embeddings_sentences = embeddings_sentences.detach().cpu().numpy()

    return embeddings_sentences, embeddings_words


def add_noise_to_embeddings(embeddings, num_augmentations, noise_std_dev=0.01):
    """

    :param embeddings: input list of embeddings generated by get_sent_word_embeddings()
    :param num_augmentations: number of noise variations to add
    :param noise_std_dev: the standard deviation of multiplicative gaussian noise to add
    :return: list of augmented embeddings and a list of their original embedding indices
    """
    augmented_embeddings, augmented_idxs = [], []

    # Add Gaussian noise to the embeddings
    for i, embedding in enumerate(embeddings):
        for _ in range(num_augmentations):
            # mean of 1, because this is multiplicative, since we want values that are 0 (or close) to stay 0
            noise = np.random.normal(1, noise_std_dev, embedding.shape)
            augmented_embedding = embedding * noise

            augmented_embeddings.append(augmented_embedding)
            augmented_idxs.append(i)

    # only return augmented values
    return augmented_embeddings, augmented_idxs