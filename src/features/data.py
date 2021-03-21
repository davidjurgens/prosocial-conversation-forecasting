import re
import sys
import collections
from datetime import datetime

import argparse
import os
import nltk
from nltk import parse
# import pycld2
import ujson as json
from textblob import TextBlob
import textstat
import pyspark
from nltk.tokenize import TreebankWordTokenizer


sys.setrecursionlimit(999999)

tokenizer = TreebankWordTokenizer()


def metrics_helper(comment):
    '''
    Input: non-top comment
    Output: features for non-top comments that are related to its replies
    '''
    if not comment['children']:
        # leaf nodes
        node_links_count = only_link_count(comment['body'])
        return {'Replies_total_number': 1,
                'Replies_max_depth': 1,
                'Replies_links_count': node_links_count,
                'Replies_sum_score': comment['score'] if comment['score'] else 0}
    else:
        # initialization
        Replies_total_number = 0
        Replies_max_depth = 0
        Replies_links_count = 0
        Replies_sum_score = 0

        # accumulated features from replies
        for child in comment['children']:
            # recursive call
            submetrics = metrics_helper(child)
            Replies_total_number += submetrics['Replies_total_number']
            Replies_max_depth = max(Replies_max_depth,
                                    submetrics['Replies_max_depth'])
            Replies_links_count += submetrics['Replies_links_count']
            Replies_sum_score += submetrics['Replies_sum_score']

        # accumulated features from the current node
        Replies_total_number += 1
        Replies_max_depth += 1
        node_links_count = only_link_count(comment['body'])
        Replies_links_count += node_links_count
        if comment['score']:
            Replies_sum_score += comment['score']

        return {'Replies_total_number': Replies_total_number,
                'Replies_max_depth': Replies_max_depth,
                'Replies_links_count': Replies_links_count,
                'Replies_sum_score': Replies_sum_score}


def metrics(top_comments):
    '''
    Input: top_level comment
    Output: features fron this top_level comment
    '''
    Replies_total_number = 0
    Replies_max_depth = 0
    Replies_links_count = 0
    Replies_sum_score = 0

    for reply in top_comments['children']:
        reply_metrics = metrics_helper(reply)
        Replies_total_number += reply_metrics['Replies_total_number']
        Replies_max_depth = max(Replies_max_depth,
                                reply_metrics['Replies_max_depth'])
        Replies_links_count += reply_metrics['Replies_links_count']
        Replies_sum_score += reply_metrics['Replies_sum_score']

    return {'Replies_total_number': Replies_total_number,
            'Replies_max_depth': Replies_max_depth,
            'Replies_links_count': Replies_links_count,
            'Replies_sum_score': Replies_sum_score}


def clean_text(text):
    text = (re.sub(r'\[([^]]+)\]\s*\([^)]+\)', r'\1', text)
              .replace("\"", '')
              .replace('"', '')
              .replace('\n', '')
              .replace('\t', '')
              .replace('\r', ''))
    num_tokens = len(tokenizer.tokenize(text))
    return text, num_tokens


def only_link_count(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)  # noqa: W791,E501,W605
    return len(urls)


def get_day_and_hour(utc_time):
    """get day, hour and weekday from utc_time

    Args:
        utc_time (int): utc time

    Returns:
        Tuple[int, int, int]: hour, day and weekday
    """
    dt = datetime.utcfromtimestamp(utc_time)
    day = dt.day
    hour = dt.hour
    weekday = dt.weekday()
    return hour, day, weekday


def time_post_to_tlc(top_comment_utc, post_utc):
    """calcualte the time post to tlc difference
    In our initial experiment, we removed the effect of month difference.
    In this new implmentation, we added the difference in month.
    Args:
        top_comment_utc (datetime.timestamp): tlc utc time
        post_utc (datetime.timestamp): post utc time

    Returns:
        int: time different in hours
    """
    pt = datetime.utcfromtimestamp(post_utc)
    tt = datetime.utcfromtimestamp(top_comment_utc)
    month_diff = tt.month - pt.month
    day_diff = (tt.day - pt.day) + 30 * month_diff
    hour_diff = tt.hour - pt.hour
    return day_diff * 24 + hour_diff


def load_data_mapper(line):
    """load the data

    Args:
        json_file_path (Path): path to the json file

    Yields:
        Tuple(string, int, string): tlc id, tlc order and comment text
    """
    post = json.loads(line)
    for tlc_order, comment in enumerate(post['children']):
        yield comment['id'], tlc_order, comment, post


def extract_fields(idx, comment, post):
    fields = {'Top_comment_id': comment['id'],
              'Top_comment_order': idx,
              'Post_created_utc': post['created_utc'],
              'Post_score': post['score'] if post['score'] else 0,
              'Top_comment_created_utc': comment['created_utc'],
              'Top_comment_subreddit': comment['subreddit'],
              'Top_comment_link_count': only_link_count(comment['body']),
              'Top_comment_score': comment['score'] if comment['score'] else 0,
              'Top_comment_author': comment['author'],
              'Post_author': post['author'],
              'Top_comment_direct_children': len(comment['children'])
              }
    (fields['Top_comment_hour'], fields['Top_comment_day'],
     fields['Top_comment_day_of_week']) = get_day_and_hour(
        fields['Top_comment_created_utc'])
    (fields['Post_hour'], fields['Post_day'],
     fields['Post_day_of_week']) = (
         get_day_and_hour(fields['Post_created_utc']))
    fields['Top_comment_and_Post_time_difference'] = (time_post_to_tlc(
        fields['Top_comment_created_utc'], fields['Post_created_utc']))
    # get additional features
    metr = metrics(comment)
    fields['Replies_total_number'] = metr['Replies_total_number']
    fields['Replies_max_depth'] = metr['Replies_max_depth']
    fields['Replies_links_count'] = metr['Replies_links_count']
    fields['Replies_sum_score'] = metr['Replies_sum_score']
    (fields['Top_comment_text'],
     fields['Top_comment_word_count']) = clean_text(comment['body'])
    fields['Post_text'], _ = clean_text(post['title'] + ' ' + post['selftext'])
    tlc_text = fields['Top_comment_text']
    sentis = TextBlob(tlc_text).sentiment
    fields['Top_comment_polarity'] = sentis.polarity
    fields['Top_comment_subjectivity'] = sentis.subjectivity
    fields['Top_comment_readability'] = textstat.flesch_kincaid_grade(tlc_text)
    return fields


def is_English(text, verbose=False):
    """check if the string is in English

    Args:
        string (str): the string to be tested
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        bool: return True if it is in English
    """
    thre = 0.8  # threshold for % of English
    _, _, details, vectors = pycld2.detect(text, returnVectors=True)
    languages, _, _, _ = list(zip(*details))

    # ok if only contains English
    if set(languages) == {'ENGLISH'}:
        return True

    if not len(vectors):
        return False
    # calculate the length of each language
    _, lengths, langs, _ = list(zip(*vectors))

    counts = collections.defaultdict(int)
    for length, lang in zip(lengths, langs):
        counts[lang] += length

    # calculate % of English
    total_length = sum(counts.values())
    percent_of_English = counts['ENGLISH'] / total_length
    
    if verbose:
        print(f'% of English: {percent_of_English}')
    if set(languages) == {'ENGLISH', 'Unknown'} and languages[0] == 'ENGLISH' and percent_of_English > thre:
        return True
    
    return False


def process_as_iterative(feature_file_path, json_file_path):
    first = True
    with open(feature_file_path, 'w') as fout:
        with open(json_file_path, 'r') as instream:
            for line in instream:
                for _, tlc_order, comment, post in load_data_mapper(line):

                    fields = extract_fields(tlc_order, comment, post)
                    # if not is_English(fields['Top_comment_text']):
                    #     continue
                    if not fields:
                        continue
                    if first:
                        keys = sorted(fields.keys())
                        sep = '\t'
                        fout.write('{}\n'.format(sep.join(keys)))
                        first = False

                    keys = sorted(fields.keys())
                    sep = '\t'

                    ordered_fields = list()
                    for key in keys:
                        ordered_fields.append(str(fields[key]))
                    print(sep.join(ordered_fields))
                    fout.write('{}\n'.format(sep.join(ordered_fields)))


def process_with_spark(input_file, output_file, num_parts):
    conf = pyspark.SparkConf().setAppName("BasicFeatureExtractor")
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)
    texts_rdd = sc.textFile(input_file).repartition(num_parts)
    df = (texts_rdd.flatMap(lambda x: load_data_mapper(x))
                   .map(lambda x: extract_fields(x[1], x[2], x[3]))
                   .filter(lambda x: is_English(x['Top_comment_text']))
                   .toDF())
    df.repartition(1).write.csv(output_file, header=True, sep='\t')


def main():
    parser = argparse.ArgumentParser(description='Arguments for extracting basic features')
    # model parameters
    parser.add_argument('--run_on_local', action='store_true',
                        help='whether to run the code on a local machine')
    parser.add_argument('--input_file', type=str, required=True,
                        help='The path to the input file.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='The path to the output file.')
    parser.add_argument('--num_executors', type=int, default=64,
                        help='The number of executors')
    args = parser.parse_args()
    # choose where to run
    if args.run_on_local:
        process_as_iterative(args.input_file, args.output_file)
    else:
        process_with_spark(args.input_file, args.output_file, args.num_executors)


main()