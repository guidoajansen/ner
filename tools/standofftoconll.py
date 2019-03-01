# A python script to turn annotated data in standoff format (brat annotation tool) to the formats expected by Stanford NER and Relation Extractor models
# - NER format based on: http://nlp.stanford.edu/software/crf-faq.html#a
# - RE format based on: http://nlp.stanford.edu/software/relationExtractor.html#training

# Usage:
# 1) Install the pycorenlp package
# 2) Go to the installation directory and Run CoreNLP server, example: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# 3) Place .ann and .txt files from brat in the location specified in DATA_DIRECTORY
# 4) Run this script

# Cross-sentence annotation is not supported

from pycorenlp import StanfordCoreNLP
import os
from os import listdir
from os.path import isfile, join

DEFAULT_OTHER_ANNO = 'O'
STANDOFF_ENTITY_PREFIX = 'T'
STANDOFF_RELATION_PREFIX = 'R'
DATA_DIRECTORY = '../data/scitodate 2/dev'
OUTPUT_DIRECTORY = '../data/scitodate 2/'
CORENLP_SERVER_ADDRESS = 'http://localhost:9000'
RELATIONS = False

NER_TRAINING_DATA_OUTPUT_PATH = join(OUTPUT_DIRECTORY, 'dev.txt')
RE_TRAINING_DATA_OUTPUT_PATH = join(OUTPUT_DIRECTORY, 're-training-data.corp')

if os.path.exists(OUTPUT_DIRECTORY):
    if os.path.exists(NER_TRAINING_DATA_OUTPUT_PATH):
        os.remove(NER_TRAINING_DATA_OUTPUT_PATH)
    if os.path.exists(RE_TRAINING_DATA_OUTPUT_PATH):
        os.remove(RE_TRAINING_DATA_OUTPUT_PATH)
else:
    os.makedirs(OUTPUT_DIRECTORY)

sentence_count = 0
nlp = StanfordCoreNLP(CORENLP_SERVER_ADDRESS)

# looping through .ann files in the data directory
ann_data_files = [f for f in listdir(DATA_DIRECTORY) if isfile(join(DATA_DIRECTORY, f)) and f.split('.')[1] == 'ann']

labels= {
    'Device': {
        'beginning': 'B-DEV',
        'inside': 'I-DEV'
    },
    'BrandName': {
        'beginning': 'B-BRAND',
        'inside': 'I-BRAND'
    },
    'Company': {
        'beginning': 'B-COM',
        'inside': 'I-COM'
    }
}

for file in ann_data_files:
    entities = []
    relations = []

    # process .ann file - place entities and relations into 2 seperate lists of tuples
    with open(join(DATA_DIRECTORY, file), 'r') as document_anno_file:
        lines = document_anno_file.readlines()
        standoff_id = 1

        for line in lines:
            standoff_line = line.split()
            if standoff_line[0][0] == STANDOFF_ENTITY_PREFIX and standoff_line[1] in ['Device', 'BrandName', 'Company']:

                entity_start = (int(standoff_line[2]))

                for idx, word in enumerate(standoff_line[4:]):
                    entity = {}
                    entity['standoff_id'] = standoff_id
                    if idx == 0:
                        entity['entity_type'] = labels[standoff_line[1]]['beginning']
                        entity['offset_start'] = entity_start
                        entity['offset_end'] = entity_start + len(word)
                    else:
                        entity['entity_type'] = labels[standoff_line[1]]['inside']
                        entity['offset_start'] = entity_start
                        entity['offset_end'] = entity_start + len(word)

                    entity['word'] = word

                    entities.append(entity)
                    # Update entity_start with word length and a space
                    entity_start += (len(word) + 1)
                    standoff_id  += 1

    # read the .ann's matching .txt file and tokenize its text using stanford corenlp
    with open(join(DATA_DIRECTORY, file.replace('.ann', '.txt')), 'r') as document_text_file:
        document_text = document_text_file.read()

    properties = {'annotators': 'ssplit', 'outputFormat': 'json'}

    output = nlp.annotate(document_text, properties={
        'annotators': 'tokenize, ssplit, pos',
        'ssplit.newlineIsSentenceBreak': 'always',
        'tokenize.options': 'normalizeParentheses=false , normalizeOtherBrackets=false',
        'outputFormat': 'json'
    })

    training_data_length = 0
    # write text and annotations into NER and RE output files
    with open(NER_TRAINING_DATA_OUTPUT_PATH, 'a') as ner_training_data, open(RE_TRAINING_DATA_OUTPUT_PATH,
                                                                             'a') as re_training_data:
        ner_training_data.write("-DOCSTART- O")
        ner_training_data.write("\n\n")

        for sentence in output['sentences']:
            entities_in_sentence = {}
            sentence_re_rows = []

            training_data_length += 1

            for token in sentence['tokens']:
                print(token)
                offset_start = int(token['characterOffsetBegin'])
                offset_end = int(token['characterOffsetEnd'])

                re_row = {}
                entity_found = False
                ner_anno = DEFAULT_OTHER_ANNO

                # searching for token in annotated entities
                for entity in entities:
                    if offset_start >= entity['offset_start'] and offset_end <= entity['offset_end']:
                        print(entity)

                        ner_anno = entity['entity_type']

                    # multi-token entities for RE need to be handled differently than NER
                    if offset_start == entity['offset_start'] and offset_end <= entity['offset_end']:
                        entities_in_sentence[entity['standoff_id']] = len(sentence_re_rows)
                        re_row['entity_type'] = entity['entity_type']
                        re_row['pos_tag'] = token['pos']
                        re_row['word'] = token['word']

                        sentence_re_rows.append(re_row)
                        entity_found = True
                        break
                    elif offset_start > entity['offset_start'] and offset_end <= entity['offset_end'] and len(
                            sentence_re_rows) > 0:
                        sentence_re_rows[-1]['pos_tag'] += '/{}'.format(token['pos'])
                        sentence_re_rows[-1]['word'] += '/{}'.format(token['word'])
                        entity_found = True
                        break

                if not entity_found:
                    re_row['entity_type'] = DEFAULT_OTHER_ANNO
                    re_row['pos_tag'] = token['pos']
                    re_row['word'] = token['word']

                    sentence_re_rows.append(re_row)

                # writing tagged tokens to NER training data
                ner_training_data.write('{}\t{}\n'.format(token['word'], ner_anno))


            sentence_count += 1
            ner_training_data.write('\n')

        ner_training_data.write('\n')

    print(training_data_length)
    print('Processed file pair: {} and {}'.format(file, file.replace('.ann', '.txt')))