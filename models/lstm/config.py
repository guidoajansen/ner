import os
from .utils import get_logger, get_trimmed_embedding_vectors, load_vocab, load_counts, get_processing_word

class Config():
    def __init__(self, load=True):
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.word_counts = load_counts(self.filename_counts)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_embedding_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

    # general config
    dir_output = "results/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_char = 100

    filename_dev = "data/conll/324/valid.txt"
    filename_test = "data/conll/324/test.txt"
    filename_train = "data/conll/324/train.txt"

    # filename_dev = "data/scitodate 2/dev.txt"
    # filename_test = "data/scitodate 2/test.txt"
    # filename_train = "data/scitodate 2/train.txt"

    # filename_dev = "data/scitodate/valid.txt"
    # filename_test = "data/scitodate/test.txt"
    # filename_train = "data/scitodate/train.txt"

    # dim_word = 200
    # embedding = "pubmed"
    # filename_embedding = "data/embeddings/bionlp/PubMed-shuffle-win-30.txt".format(dim_word)
    # filename_trimmed = "data/embeddings/bionlp/PubMed-shuffle-win-30.trimmed.npz".format(dim_word)

    dim_word = 300
    embedding = 'glove'
    filename_embedding = "data/embeddings/glove/glove.6B.{}d.txt".format(dim_word)
    filename_trimmed = "data/embeddings/glove/glove.6B.{}d.trimmed.npz".format(dim_word)

    # filename_dev = filename_test = filename_train = "data/test.txt"  # test

    use_pretrained = False
    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"
    filename_counts = "data/counts.txt"

    # training
    train_embeddings = True
    embedding = "custom"

    nepochs = 30
    dropout = 0.5
    batch_size = 10
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 15

    # attention specific
    num_heads = 8
    attention_size = None
    is_train = False

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
