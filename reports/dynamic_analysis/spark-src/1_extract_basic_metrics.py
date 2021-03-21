"""Data pipeline for extracting basic metrics"""
import sys
import random
import ast
import re

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


def inject_one_depth(node):
    res = 1
    for subnode in node['children']:
        inject_one_depth(subnode)  # inplace
        res = max(res, subnode['replies_depth'])
    node['replies_depth'] = res


def inject_depths(trlst):
    for child in trlst:
        inject_one_depth(child)
    return trlst


def flat_one_tree(node):
    res = list()
    for child in node['children']:
        res += flat_one_tree(child)
    node.pop('children')
    return res + [node]


def flat_trees(trlst):
    res = list()
    for child in trlst:
        res += flat_one_tree(child)
    return res


def sort_tree_by_time(trlst):
    trlst.sort(key=lambda x: x['created_utc'])
    return trlst


def split_tree_by_time(trlst):
    assert len(trlst) % 2 == 0
    tl = len(trlst) // 2
    return trlst[:tl], trlst[tl:]


def process_text(text):
    tlcid, rply_list = ast.literal_eval(text)
    inject_depths(rply_list)
    res = flat_trees(rply_list)
    return tlcid, res


def process_tree(input_tuples):
    tlcid, tree = input_tuples
    p0, p1 = split_tree_by_time(sort_tree_by_time(tree))
    return [(f'{tlcid}-0', p0), (f'{tlcid}-1', p1)]


def find(text):
    """count the number of links in the raw text
    Args:
        text ([type]): [description]
    Returns:
        [type]: [description]
    """
    exp = (r"http[s]?://"
           r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|"
           r"(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    urls = re.findall(exp, text)
    return len(urls)


def extract_features(input_tuples):
    tlcid, rpy_list = input_tuples
    link_count, total_c_count, direct_c_count = 0, 0, 0
    score_sum = 0
    depth = 0
    for rpy in rpy_list:
        link_count += find(rpy['body'])
        total_c_count += 1
        score_sum += rpy['score']
        depth = max(depth, rpy['replies_depth'])
        if rpy['parent_id'][3:] == tlcid[:-2]:
            direct_c_count += 1
    return tlcid, link_count, total_c_count, direct_c_count, score_sum, depth


sc = SparkContext.getOrCreate()
tlcs_raw = sc.textFile("research-out/sampled_tlcs_ok/part-00*")
flatten_tlcs = tlcs_raw.map(process_text)
spiltted_tlcs = flatten_tlcs.flatMap(process_tree)
# spiltted_tlcs.saveAsTextFile("research-out/splitted_sampled_tlcs")

features_rdd = spiltted_tlcs.map(extract_features)
features_rdd0_fid = features_rdd.filter(lambda x: x[0].endswith('-0'))
features_rdd1_fid = features_rdd.filter(lambda x: x[0].endswith('-1'))

features_rdd0 = features_rdd0_fid.map(lambda x: (x[0][:-2], *x[1:]))
features_rdd1 = features_rdd1_fid.map(lambda x: (x[0][:-2], *x[1:]))


headers = ['Top_comment_id', 'Replies_links_count', 'Replies_total_number',
           'Top_comment_direct_children', 'Replies_sum_score',
           'Replies_max_depth']
df0 = features_rdd0.toDF(headers)

df1 = features_rdd1.toDF(headers)

df0.repartition(1).write.csv("research-out/basic_features_0",
                             header=True, sep='\t')


def get_total_direct_children(input_tuples):
    tlcid, rpy_list = input_tuples
    direct_c_count = 0
    for rpy in rpy_list:
        if rpy['parent_id'][3:] == tlcid:
            direct_c_count += 1
    return f'{tlcid}-1', direct_c_count


p1_all_direct_c = flatten_tlcs.map(get_total_direct_children)
total_direct_df = p1_all_direct_c.toDF(['Top_comment_id',
                                        'Top_comment_direct_children'])

df1_fixed = df1.drop('Top_comment_direct_children').join(
    total_direct_df,
    ['Top_comment_id'],
    'inner'
)

df1_fixed.repartition(1).write.csv("research-out/basic_features_1",
                                   header=True, sep='\t')

# df0.repartition(1).write.csv("data_for_dynamic_analysis/basic_features_0.csv",
#                              header=True, sep='\t')
# df1_fixed.repartition(1).write.csv("data_for_dynamic_analysis/basic_features_1.csv",
#                              header=True, sep='\t')
