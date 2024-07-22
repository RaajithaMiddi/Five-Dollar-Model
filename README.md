# Forked Five-Dollar Model

Modification to this project for GeorgiaTech DL final project. Includes using novel datasets for new sprite generation and experiments of model architecture or configuration.


## The Five-Dollar Model: Generating Game Maps and Sprites from Sentence Embeddings

### Timothy Merino, Roman Negri, Dipika Rajesh, M Charity, Julian Togelius

The five-dollar model is a lightweight text-to-image generative architecture that generates low dimensional images or tile maps from an encoded text prompt. This model can successfully generate accurate and aesthetically pleasing content in low dimensional domains, with limited amounts of training data. Despite the small size of both the model and datasets, the generated images or maps are still able to maintain the encoded semantic meaning of the textual prompt. We apply this model to three small datasets: pixel art video game maps, video game sprite images, and down-scaled emoji images and apply novel augmentation strategies to improve the performance of our model on these limited datasets. We evaluate our models’ performance using cosine similarity score between text-image pairs generated by the CLIP VIT-B/32 model to demonstrate quality generation.

Read the full paper at https://arxiv.org/abs/2308.04052 .

![image](https://github.com/TimMerino1710/five-dollar-model/assets/83784750/ce4358ef-88b9-455f-8e3f-8e03825b7b8d)

Soon to be published in AIIDE-2023.

For any questions about the paper, code, or datasets you can contact us at:

Tim - tm3477 [at] nyu.edu

Roman - rvn9303 [at] nyu.edu
