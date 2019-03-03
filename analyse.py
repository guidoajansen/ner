import pandas as pd

from models.lstm.config import Config
from models.lstm.utils import CoNLLDataset

def main():
    config = Config()

    dataset = CoNLLDataset("data/conll/1246/train.txt", config.processing_word,
                         config.processing_tag, config.max_iter)

    # dataset = CoNLLDataset("data/scitodate 2/train.txt", config.processing_word,
    #                      config.processing_tag, config.max_iter)

    tags = { idx: tag for tag, idx in config.vocab_tags.items() }
    words = { idx: tag for tag, idx in config.vocab_words.items() }
    counts = config.word_counts

    # samples = 324
    #
    # dataset = dataset.sample(samples)

    data = []

    for sample in dataset:

        raw = []

        # Sentence length
        length = len(sample[0])
        raw.append(length)

        # Tokens
        tokens = [ words[token[-1]] for token in sample[0] ]
        raw.append(tokens)

        # Unique
        unique_tokens = []
        unique = 0
        for token in tokens:
            if token == "$UNK$" or token == "$NUM$":
                continue

            if counts[token] == "1":
                unique += 1
                unique_tokens.append(token)

        raw.append(unique)

        # Average Token Length
        avg_token_len = sum(map(len, tokens)) / len(tokens)
        raw.append(avg_token_len)

        # Entities
        entities = []
        unique = 0
        unique_tokens = []
        single_tag_counter = [0 for i in range(len(tags))]

        for idx, tag in enumerate(sample[1]):
            if tags[tag] != "O":
                word = words[sample[0][idx][-1]]
                entities.append(word)
                if word == "$UNK$" or word == "$NUM$":
                    continue

                if counts[word] == "1":
                    unique += 1
                    unique_tokens.append(word)

            single_tag_counter[tag] += 1

        raw.append(entities)
        raw.append(unique)

        # Average Entity Length
        if len(entities) > 0:
            avg_entity_len = sum(map(len, entities)) / len(entities)
            raw.append(avg_entity_len)
        else:
            raw.append(-1)

        # Density
        density = len(entities) / length * 100
        raw.append(density)

        # Append Single Entities in Order
        raw.append(single_tag_counter[0]) # B-PER
        raw.append(single_tag_counter[2]) # I-PER
        raw.append(single_tag_counter[6]) # B-LOC
        raw.append(single_tag_counter[3]) # I-LOC
        raw.append(single_tag_counter[4]) # B-ORG
        raw.append(single_tag_counter[8]) # I-ORG
        raw.append(single_tag_counter[7]) # B-MISC
        raw.append(single_tag_counter[5]) # I-MISC
        raw.append(single_tag_counter[1]) # O


        # # Append Single Entities in Order
        # raw.append(single_tag_counter[0]) # B-COM
        # raw.append(single_tag_counter[3]) # I-COM
        # raw.append(single_tag_counter[2]) # B-BRAND
        # raw.append(single_tag_counter[4]) # I-BRAND
        # raw.append(single_tag_counter[5]) # B-DEV
        # raw.append(single_tag_counter[6]) # I-DEV
        # raw.append(single_tag_counter[1]) # O

        data.append(raw)

    analysis = pd.DataFrame(data=data, columns=["Length", "Tokens", "Unique Tokens", "Avg Token Length", "Entities", "Unique Entities", "Avg Entity Length", "Density", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"])
    # analysis = pd.DataFrame(data=data, columns=["Length", "Tokens", "Unique Tokens", "Avg Token Length", "Entities", "Unique Entities", "Avg Entity Length", "Density", "B-COM", "I-COM", "B-BRAND", "I-BRAND", "B-DEV", "I-DEV", "O"])

    analysis.to_csv("data/analysis/conll1246.tsv", sep='\t')
    # analysis.to_csv("data/analysis/scito1246.tsv", sep='\t')
    # analysis.to_csv("data/analysis/pubmed.tsv", sep='\t')


if __name__ == "__main__":
    main()
