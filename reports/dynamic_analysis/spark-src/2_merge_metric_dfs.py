"""Data pipeline for extracting basic metrics"""
import sys
import random
from functools import reduce

# this code allows you to run the following code locally
import findspark
findspark.init()
import pyspark  # noqa: E402

sc = pyspark.SparkContext(appName="hw")
# the code is run on the Cavium cluster of UMICH ARC-TS
# Some libraries are pre-loaded as part of the cluster configuration
#   e.g. SparkContext
# If you are to run this code on a local machine,
# pay attention to those libraries: https://arc-ts.umich.edu/cavium/user-guide/
# or contact the authors

random.seed(42)

sys.setrecursionlimit(999999)


def fixed_eform_df(data, headers):
    fixed_h0 = ['Top_comment_id'] + [f'{x}_first_half' for x in headers]
    fixed_h1 = ['Top_comment_id'] + [f'{x}_second_half' for x in headers]
    data0 = data.select(fixed_h0)
    data1 = data.select(fixed_h1)
    fixed_h = ['Top_comment_id'] + headers
    res = list()
    for d, h in [(data0, fixed_h0), (data1, fixed_h1)]:
        tmp = reduce(lambda data, idx: data.withColumnRenamed(h[idx], fixed_h[idx]), range(len(h)), d)
        res.append(tmp)
    return res[0], res[1]


# it may look weird that spark is not imported anywhere..
# spark is preloaded on the Cavium cluster of UMICH ARC-TS
sp_df = spark.read.load('data_for_dynamic_analysis/support_politeness.tsv',
                        format="csv", sep="\t", inferSchema="true",
                        header="true")

jef_names2 = [
    'Replies_average_politeness', 'Replies_average_support'
]

sp_df0, sp_df1 = fixed_eform_df(sp_df, jef_names2)


toxc_df = spark.read.load('data_for_dynamic_analysis/toxicity_children_count.tsv',
                          format="csv", sep="\t", inferSchema="true",
                          header="true")


jef_names1 = [
    'Replies_untuned_toxicity_children_count',
    'Replies_tuned_toxicity_children_count',
    'Replies_tuned_toxicity_children_count',
    'Replies_untuned_non_toxic_percentage',
    'Replies_tuned_non_toxic_percentage',
]

toxc_df0, toxc_df1 = fixed_eform_df(toxc_df, jef_names1)

# it may look weird that spark is not imported anywhere..
# spark is preloaded on the Cavium cluster of UMICH ARC-TS

accd_df0 = spark.read.load('data_for_dynamic_analysis/twelve_mimicry_metrics0.tsv',
                          format="csv", sep="\t", inferSchema="true",
                          header="true")

accd_df1 = spark.read.load('data_for_dynamic_analysis/twelve_mimicry_metrics1.tsv',
                          format="csv", sep="\t", inferSchema="true",
                          header="true")


def change_column_name(d, old_h, fixed_h):
    tmp = reduce(lambda data, idx: data.withColumnRenamed(old_h[idx], fixed_h[idx]), range(len(fixed_h)), d)
    return tmp


jsf_old_names = [
    'Top_comment_id', 'num_informative_replies', 'num_advice_replies',
    'num_laughter_replies', 'num_gratitude_replies',
    'num_fundraising_URL_replies', 'num_informative_URL_replies',
    'num_i_language_replies', 'num_compliments'
    ]

jsf_new_names = [
    'Top_comment_id', 'Replies_informative_count', 'Replies_advice_count',
    'Replies_laughter_count', 'Replies_gratitude_count',
    'Replies_fundraising_URL_count',
    'Replies_informative_URL_count', 'Replies_i_language_count',
    'Replies_compliments_count'
]

jgs_df0_ename = spark.read.load('data_for_dynamic_analysis/jurgens-prosocial-metrics.first-half.tsv',
                                format="csv", sep="\t", inferSchema="true",
                                header="true")

jgs_df1_ename = spark.read.load('data_for_dynamic_analysis/jurgens-prosocial-metrics.second-half.tsv',
                                format="csv", sep="\t", inferSchema="true",
                                header="true")

jgs_df0 = change_column_name(jgs_df0_ename, jsf_old_names, jsf_new_names)
jgs_df1 = change_column_name(jgs_df1_ename, jsf_old_names, jsf_new_names)


p_sustained_df0 = spark.read.load('data_for_dynamic_analysis/yiming/first_half/Replies_distinct_pairs_of_sustained_conversation.tsv',
                                  format="csv", sep="\t", inferSchema="true",
                                  header="true")

p_sustained_df1 = spark.read.load('data_for_dynamic_analysis/yiming/second_half/Replies_distinct_pairs_of_sustained_conversation.tsv',
                                  format="csv", sep="\t", inferSchema="true",
                                  header="true")

max_turns_df0 = spark.read.load('data_for_dynamic_analysis/yiming/first_half/Replies_max_turns_of_sustained_conversations.tsv',
                                format="csv", sep="\t", inferSchema="true",
                                header="true")
max_turns_df1 = spark.read.load('data_for_dynamic_analysis/yiming/second_half/Replies_max_turns_of_sustained_conversations.tsv',
                                format="csv", sep="\t", inferSchema="true",
                                header="true")

basic_df0 = spark.read.load('data_for_dynamic_analysis/basic_features_0.csv',
                            format="csv", sep="\t", inferSchema="true",
                            header="true")

basic_df1 = spark.read.load('data_for_dynamic_analysis/basic_features_1.csv',
                            format="csv", sep="\t", inferSchema="true",
                            header="true")


sp_df0 = sp_df0.dropDuplicates(["Top_comment_id"])
toxc_df0 = toxc_df0.dropDuplicates(["Top_comment_id"])

targets0 = [jgs_df0, p_sustained_df0, accd_df0, sp_df0, toxc_df0, 
            max_turns_df0]
res_df0 = basic_df0


for x in targets0:
    res_df0 = res_df0.join(
        x,
        ['Top_comment_id'],
        'inner'
    )

sp_df1 = sp_df1.dropDuplicates(["Top_comment_id"])
toxc_df1 = toxc_df1.dropDuplicates(["Top_comment_id"])


targets1 = [jgs_df1, p_sustained_df1, accd_df1, sp_df1, toxc_df1, 
            max_turns_df1]
res_df1 = basic_df0


for x in targets1:
    res_df1 = res_df1.join(
        x,
        ['Top_comment_id'],
        'inner'
    )


metrics = [
    'Top_comment_id',
    'Replies_informative_count',
    'Replies_links_count',
    'Replies_max_depth',
    'Replies_sum_score',
    'Replies_total_number',
    'Top_comment_article_accommodation',
    'Top_comment_certain_accommodation',
    'Top_comment_conj_accommodation',
    'Top_comment_discrep_accommodation',
    'Top_comment_excl_accommodation',
    'Top_comment_incl_accommodation',
    'Top_comment_ipron_accommodation',
    'Top_comment_negate_accommodation',
    'Top_comment_quant_accommodation',
    'Top_comment_tentat_accommodation',
    'Replies_advice_count',
    'Replies_laughter_count',
    'Replies_gratitude_count',
    'Replies_informative_URL_count',
    'Replies_i_language_count',
    'Replies_compliments_count',
    'Replies_untuned_toxicity_children_count',
    'Top_comment_direct_children',
    'Replies_distinct_pairs_of_sustained_conversation',
    'Replies_max_turns_of_sustained_conversations',
    'Replies_untuned_non_toxic_percentage'
]  # 26 metrics


pca_df0 = res_df0.select(metrics)
pca_df1 = res_df1.select(metrics)

pca_df0.repartition(1).write.csv("research-out/pca_df0",
                                 header=True, sep='\t')
pca_df1.repartition(1).write.csv("research-out/pca_df1",
                                 header=True, sep='\t')
