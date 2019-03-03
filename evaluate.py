from models.lstm.utils import CoNLLDataset
from models.lstm.model import Model
from models.lstm.config import Config

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds, attn = model.predict(words_raw)
        corr_labels = []

        for idx, word in enumerate(words_raw):
            # model.logger.info(word + '\t\t\t' + preds[idx])
            corr_labels.append(word + '  (' + preds[idx] + ')')

        if len(words_raw) > 1:

            attn_matrix = pd.DataFrame(attn[7], index=corr_labels, columns=corr_labels)

            f, ax = plt.subplots(figsize=(20, 20))
            corr = attn_matrix.corr()
            ax.xaxis.set_ticks_position('top')

            sns.set(font_scale=.6)

            sns.heatmap(corr, cmap="Blues", ax=ax)

            plt.show()

        else:
            print("Can not show correlations for single word sentences")

        # to_print = align_data({"input": words_raw, "output": preds})

        # for key, seq in to_print.items():
        #     model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    print(len(test))

    # build model
    model = Model(config)
    model.build()
    model.restore_session(config.dir_model)



    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
