# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# sentences = [
#     "That is a happy person",
#     "That is a happy dog",
#     "That is a very happy person",
#     "Today is a sunny day"
# ]
# embeddings = model.encode(sentences)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities.shape)
# # [4, 4]


# ----------------------------------------------------

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])