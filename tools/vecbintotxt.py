from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('../data/embeddings/bionlp/PubMed-shuffle-win-30.bin', binary=True)
model.save_word2vec_format('../data/embeddings/bionlp/PubMed-shuffle-win-30.txt', binary=False)
