from models.lstm.config import Config
from models.lstm.utils import CoNLLDataset

from models.lstm.model import Model


def main():
    # create instance of config
    config = Config()

    # build model
    model = Model(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    print(len(dev))
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    print(len(train))

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
