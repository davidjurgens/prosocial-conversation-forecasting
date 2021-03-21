import sys
import os
import gzip
from collections import Counter

import argparse
import pyspark
import string
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

letter_set = set(string.ascii_uppercase + string.ascii_lowercase)
punc_set = {"'", '-'}
valid_set = letter_set | punc_set


def parse_column_file(file_name):
    """parse the wet files
    Args:
        file_name (string): a list of file names
    Returns:
        list[list[string]]: a list of documents
    """


def val_and_std(token):
    """check the validity of a token; standard it if needed
    Args:
        token (string): a token
    Returns:
        string: an standardized token if the token is valid
        otherwise return None
    """
    token = token.strip("\"\'-")
    tran_t = token.replace("-", "").replace("'", "")
    if tran_t.isascii() and tran_t.isalpha():
        return token.lower()
    return None


def count_and_serialize(doc, v2id_bct):
    """format a doc
    Args:
        doc (dict[string, int]): a doc
        v2id_bct (dict(string, int)): vocab dictionary
    Returns:
        string: the serialized document
    """
    v2id = v2id_bct.value
    return ' '.join([f'{v2id[k]}:{v}' for k, v in doc.items()])


def word_doc_freq_map(doc):
    """emit words
    Args:
        doc (list[string]): one document
    Returns:
        zip(string, int): emitted words
    """
    words = set(doc)
    return zip(words, [1] * len(words))


def read_and_clean_records(args):
    """
    read and clean records from compressed dumps
    Returns:
        a persisted rdd with valid docs
    """
    # read wet file names
    # set up for file I/O
    # read wet files

    sp_df = spark.read.load(args.input_file,
                           format="csv", sep="\t", inferSchema="true",
                           header="true")
 
    docs_rdd_with_st = (sp_df.select(['Top_comment_id', 'Top_comment_text', 'Post_text'])
                        .rdd
                        .repartition(args.num_executors)
                        .map(lambda pld: (pld[0], [val_and_std(t) for t in tokenizer.tokenize(pld[1])]))
                        .map(lambda doc: (doc[0], [x for x in doc[1] if x]))
                        .filter(lambda doc: len(doc[1]) > 10)
                        .persist(pyspark.StorageLevel.DISK_ONLY))

    # collect words to keep by their document frequencies
    old_n_docs = docs_rdd_with_st.count()
    high_freq = old_n_docs * 0.9
    words_to_keep = set(docs_rdd_with_st
                        .map(lambda x: x[1])
                        .flatMap(lambda doc: word_doc_freq_map(doc))
                        .reduceByKey(lambda x, y: x + y)
                        .map(lambda x: (x[0], x[1]))
                        .filter(lambda x:
                                args.low_freq_threshold <= x[1] < high_freq)
                        .map(lambda x: x[0]).collect())
    # remove stop words
    with open(args.stop_words_file, 'r') as istream:
        stop_words = set()
        for line in istream.readlines():
            stop_words.add(line.strip())
    words_to_keep_bct = sc.broadcast(words_to_keep - stop_words)
    doc_rdd = (docs_rdd_with_st.map(lambda doc: (doc[0],
                                    [t for t in doc[1]
                                     if t in words_to_keep_bct.value]))
                               .filter(lambda doc: len(doc[1]) > 0)
                               .map(lambda doc: (doc[0], ' '.join(doc[1])))
                               .persist(pyspark.StorageLevel.DISK_ONLY))
    return doc_rdd


if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser(description='Arguments for process topic modeling data')
    # model parameters
    parser.add_argument('--input_file', type=str, required=True,
                        help=('The path to the dataframe (tsv from data.py).'
                        'e.g. research-out/train/part-00000-c824c2b0-19d6-4eaf-a6f7-efd82e586274-c000.csv'))
    parser.add_argument('--output_file', type=str, required=True,
                        help='The path to the output file.')
    parser.add_argument('--low_freq_threshold', type=int, required=False,
                        default=10,
                        help='the words need to appear in at least such number of docs')
    parser.add_argument('--num_executors', type=int, default=64,
                        help='The number of executors')
    parser.add_argument('--stop_words_file', type=str, required=True,
                        default='models/english', help="path to stopword file")
    args = parser.parse_args()

    # setup SparkContext
    conf = pyspark.SparkConf().setAppName("TopicModelingProcessor")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    doc_rdd = read_and_clean_records(args)
    (doc_rdd.toDF(['Top_comment_id', 'text_for_topic_modeling'])
            .repartition(1)
            .write.csv(args.output_file, header=True, sep='\t'))
    # close SparkContext
    sc.stop()
