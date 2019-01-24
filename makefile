glove:
	wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip"
	unzip ./data/embeddings/glove.6B.zip -d data/glove.6B/
	rm ./data/glove.6B.zip

build:
	python build.py

run:
	python train.py
	python evaluate.py
