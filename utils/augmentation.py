import numpy as np


def generate_mixup(imgs, embeddings, n_mixups=1, lmbda=0.5):
    """
    Applies mixup by randomly picking another image and soft updating the image and embedding
    to produce a new sample.

    :param imgs: raw pre OHT images
    :param embeddings: corresponding embedding vectors
    :param n_mixups: number of random observations to mix up
    :param lmbda: soft update parameter
    :return: augmented images, embeddings, and original idxs
    """
    augmented_imgs, augmented_embeddings, augmented_idxs = [], [], []
    n_embeddings = len(embeddings)

    for i, (img, embedding) in enumerate(zip(imgs, embeddings)):
        # Randomly select n_mixups indices without replacement
        mixup_indices = np.random.choice(n_embeddings, n_mixups, replace=False)

        # for each sampled index...
        for idx in mixup_indices:
            # make sure we're logging what was picked for debugging purposes
            augmented_idxs.append((i, idx))

            # Get the corresponding levels, labels, and embeddings
            mix_img = imgs[idx]
            mix_embedding = embeddings[idx]

            # Interpolate the levels, labels, and embeddings
            augmented_img = lmbda * img + (1 - lmbda) * mix_img
            augmented_embedding = lmbda * embedding + (1 - lmbda) * mix_embedding

            # Append the new data to the augmented lists
            augmented_imgs.append(augmented_img)
            augmented_embeddings.append(augmented_embedding)

    # only return the augmented values
    return np.asarray(augmented_imgs), np.asarray(augmented_embeddings), augmented_idxs


def interpolate_embeddings(embeddings, alt_embeddings, n_steps=1):
    """
    A different form of mixup where we take the original embedding and the GPT embedding and
    generate additional embeddings in between via interpolation.

    The original method randomly picked a label but that's probably not useful here?

    :param embeddings: list of original embeddings
    :param alt_embeddings: list of GPT embeddings
    :param n_steps: number interpolated samples to draw
    :return: list of interpolated embeddings and list of original indices
    """
    interpolated_embeddings, interpolated_idxs = [], []
    n_embeddings = len(embeddings)

    for i in range(n_embeddings):
        alpha_values = np.linspace(0, 1, n_steps + 2)[1:-1]  # Exclude the 0 and 1 values

        for alpha in alpha_values:
            interpolated_embedding = alpha * embeddings[i] + (1 - alpha) * alt_embeddings[i]
            interpolated_embeddings.append(interpolated_embedding)
            interpolated_idxs.append(i)

    # only return the augmented samples
    return interpolated_embeddings, interpolated_idxs