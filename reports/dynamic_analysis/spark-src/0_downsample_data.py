"""Data pipeline for downsampling data for dynamic analysis"""

import json
import sys
import random

# this code allows you to run the following code locally
import findspark
findspark.init()
import pyspark  # noqa: E402
from pyspark import SparkContext  # noqa: E402

sc = pyspark.SparkContext(appName="hw")
# the code is run on the Cavium cluster of UMICH ARC-TS
# Some libraries are pre-loaded as part of the cluster configuration
#   e.g. SparkContext
# If you are to run this code on a local machine,
# pay attention to those libraries: https://arc-ts.umich.edu/cavium/user-guide/
# or contact the authors

random.seed(42)
sys.setrecursionlimit(999999)


def tlc_replies_mapper(line):
    post = json.loads(line)
    res = list()
    for idx, comment in enumerate(post['children']):
        key = comment['subreddit'] + "\t" + comment['subreddit_id'] + \
            "\t" + comment['id']
        value = comment['children']
        res.append((key, value,))
    return res


def replies_count_helper(replies):
    res = 0
    # print(replies)
    for root in replies:
        res += replies_count_helper(root['children']) + 1
    return res


def replies_count_mapper(input_tuples):
    k, replies = input_tuples
    res = replies_count_helper(replies)
    return (k, res)


def ids_per_srd_freq_mapper(input_tuples):
    # assert len(input_tuples) == 2, input_tuples
    k, v = input_tuples
    srd, srd_id, tlcid = k.split('\t')
    return (f'{srd}\t{srd_id}\t{v}', tlcid)


def sample_tlc(input_tuples):
    k, tlcs = input_tuples
    # k -> srd + srd_id + freq
    res = tlcs.split('\t')
    nsample = 100
    if len(res) > nsample:
        return k, random.choices(res, k=nsample)
    return k, res


sc = SparkContext.getOrCreate()
lines = sc.textFile("research-data/filtered.dev.json")
tlcs = lines.flatMap(tlc_replies_mapper)
tlcs.saveAsTextFile("research-out/whole_tlcs")

counts = tlcs.map(replies_count_mapper).cache()
freq = counts.reduceByKey(lambda x, y: x + y)
# freq.saveAsTextFile("research-out/whole_num_replies_per_tlc")
# -> (subreddit + subreddit_id + id, freq)

even_subreddit = freq.filter(lambda x: x[1] > 0 and x[1] % 2 == 0)
# even_subreddit.saveAsTextFile("research-out/even_num_replies_per_tlc")

valid_tlc = even_subreddit.map(ids_per_srd_freq_mapper).cache()
tlc_list_per_freq = valid_tlc.reduceByKey(lambda x, y: x + '\t' + y)
# tlc_list_per_freq.saveAsTextFile("research-out/tlc_pool")

nsample = 50
unique_tlc = tlc_list_per_freq.filter(
    lambda x: len(x[1].split('\t')) >= nsample
)
sampled_from_unique_tlc = unique_tlc.map(
    sample_tlc
)

sampled_from_unique_tlc.saveAsTextFile("research-out/sampled_pool")
# # (srd + srd_id + freq, tlc_id)
flat_samples = sampled_from_unique_tlc.flatMap(
    lambda l: [(l[0], value) for value in l[1]]
)

discrete_samples = flat_samples.map(lambda x: (*(x[0].split('\t')), x[1]))
sample_df = discrete_samples.toDF(
    ['Top_comment_subreddit', 'Top_comment_subreddit_id', 'n_replies',
     'Top_comment_id']
)

# it may look weird that spark is not imported anywhere..
# spark is preloaded on the Cavium cluster of UMICH ARC-TS
full_df = spark.read.load('research-data/test.tsv',
                          format="csv", sep="\t", inferSchema="true",
                          header="true")

two_cols_from_sample = sample_df.select('Top_comment_id', 'n_replies')

res_df = two_cols_from_sample.join(
            full_df,
            ['Top_comment_id'],
            'inner'
        ).cache()

res_df.repartition(1).write.csv("research-out/test_sampled_df",
                                header=True, sep='\t')
