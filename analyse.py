import pandas as pd

from models.lstm.config import Config
from models.lstm.utils import CoNLLDataset

def main():
    config = Config()

    # dataset = CoNLLDataset("data/conll/train.txt", config.processing_word,
    #                      config.processing_tag, config.max_iter)

    dataset = CoNLLDataset("data/scitodate/train.txt", config.processing_word,
                         config.processing_tag, config.max_iter)

    tags = { idx: tag for tag, idx in config.vocab_tags.items() }
    words = { idx: tag for tag, idx in config.vocab_words.items() }
    # counts = config.count_words
    # print(counts)

    print(config.vocab_words)

    dataset = dataset.sample(324)

    data = []

    for sample in dataset:

        raw = []

        # Sentence length
        length = len(sample[0])
        raw.append(length)

        # Tokens
        tokens = [ words[token[-1]] for token in sample[0] ]
        raw.append(tokens)

        # Average Token Length
        avg_token_len = sum(map(len, tokens)) / len(tokens)
        raw.append(avg_token_len)

        # Entities
        entities = []
        single_tag_counter = [0 for i in range(len(tags))]

        for idx, tag in enumerate(sample[1]):
            if tag != 4:
                entities.append(words[sample[0][idx][-1]])

            single_tag_counter[tag] += 1

        raw.append(entities)

        # Average Entity Length
        if len(entities) > 0:
            avg_entity_len = sum(map(len, entities)) / len(entities)
            raw.append(avg_entity_len)
        else:
            raw.append(-1)

        # Density
        density = len(entities) / length * 100
        raw.append(density)

        # # Append Single Entities in Order
        # raw.append(single_tag_counter[2]) # B-PER
        # raw.append(single_tag_counter[5]) # I-PER
        # raw.append(single_tag_counter[8]) # B-LOC
        # raw.append(single_tag_counter[6]) # I-LOC
        # raw.append(single_tag_counter[4]) # B-ORG
        # raw.append(single_tag_counter[1]) # I-ORG
        # raw.append(single_tag_counter[3]) # B-MISC
        # raw.append(single_tag_counter[0]) # I-MISC
        # raw.append(single_tag_counter[7]) # O

        # Append Single Entities in Order
        raw.append(single_tag_counter[1]) # B-COM
        raw.append(single_tag_counter[0]) # I-COM
        raw.append(single_tag_counter[5]) # B-BRAND
        raw.append(single_tag_counter[2]) # I-BRAND
        raw.append(single_tag_counter[3]) # B-DEV
        raw.append(single_tag_counter[6]) # I-DEV
        raw.append(single_tag_counter[4]) # O

        data.append(raw)

    # analysis = pd.DataFrame(data=data, columns=["Length", "Tokens", "Avg Token Length", "Entities", "Avg Entity Length", "Density", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"])
    analysis = pd.DataFrame(data=data, columns=["Length", "Tokens", "Avg Token Length", "Entities", "Avg Entity Length", "Density", "B-COM", "I-COM", "B-BRAND", "I-BRAND", "B-DEV", "I-DEV", "O"])

    print(analysis.head())

    # analysis.to_csv("data/analysis/conll.tsv", sep='\t')
    analysis.to_csv("data/analysis/scito.tsv", sep='\t')


if __name__ == "__main__":
    main()
