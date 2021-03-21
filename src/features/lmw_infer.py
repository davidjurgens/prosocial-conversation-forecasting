import argparse
import os
from scipy import spatial
import pandas as pd
import little_mallet_wrapper as lmw





def infer_lmw(tmp_dir, infer_df, text_type, path_to_mallet, split, path_to_model):
    text_col = f'{text_type}_text'
    path_to_infer_data           = os.path.join(tmp_dir, f'{split}.{text_col}.split.txt')
    path_to_formatted_infer_data = os.path.join(tmp_dir, f'mallet.{split}.{text_col}')
    path_to_store_result_distributions = os.path.join(tmp_dir, f'mallet.topic_distributions.{split}.{text_col}')

    infer_data = [lmw.process_string(t).strip() for t in infer_df[text_col]]
    infer_data = [d for d in infer_data if d.strip()]

    lmw.import_data(path_to_mallet,
                    path_to_infer_data,
                    path_to_formatted_infer_data,
                    infer_data)

    lmw.infer_topics(path_to_mallet,
                     path_to_model,
                     path_to_formatted_infer_data,
                     path_to_store_result_distributions)
    dist_data = lmw.load_topic_distributions(path_to_store_result_distributions)
    num_topics = len(dist_data[0])
    headers = [f'{text_type}_topic{x}' for x in range(num_topics)]
    tpc_df = pd.DataFrame(dist_data, columns=headers)
    return dist_data, tpc_df


def cal_topic_cosine_similarity(data1, data2):
    sims = list()
    for x, y in zip(data1, data2):
        sims.append(1 - spatial.distance.cosine(x, y))
    return pd.DataFrame(sims, columns=['Post_Top_comment_topic_cosine_similarity'])


def cal_post_tlc_sim():
    """calculate the similarity between post and tlc topics
    note that the dataframe has to fit in memory
    """
    # get arguments
    parser = argparse.ArgumentParser(description='Arguments for Eigenmetric Regression')
    parser.add_argument('--infer_df_path', type=str, required=True, help=('path to the training data'
                        'e.g. part-00000-7c30c29e-bc47-4069-8b52-6e8ee75a429d-c000.csv'))
    parser.add_argument('--split', type=str, required=True, help=('a name for the split, '
                        'e.g. dev, test..., used to store intermediate files'))
    parser.add_argument('--path_to_mallet', help='path to the mallet module, e.g. mallet-2.0.8/bin/mallet')
    parser.add_argument('--path_to_model', help='path to model checkpoint from lmw_train.py, e.g. models/lmw-output/mallet.model.20')
    parser.add_argument('--tmp_dir', type=str, required=True,
                        help='directory to save intermediate files, e.g. data/interim')
    parser.add_argument('--path_to_store_res_df', type=str, required=True,
                        help='path to store resulting dataframe')

    parser.add_argument()
    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    # get dataframe
    infer_df = pd.read_csv(args.infer_df_path, sep='\t', engine='python')
    # calculate lmw distribution
    top_comment_dist_data, top_comment_dist_df = infer_lmw(args.tmp_dir, infer_df, 'Top_comment', 
                                                           args.path_to_mallet, args.split, args.path_to_model)
    post_dist_data, post_dist_df = infer_lmw(args.tmp_dir, infer_df, 'Post', 
                                             args.path_to_mallet, args.split, args.path_to_model)
    sim_df = cal_topic_cosine_similarity(top_comment_dist_data, post_dist_data)
    res_df = pd.concat((infer_df, top_comment_dist_df, post_dist_df, sim_df), axis=1)
    res_df.to_csv(args.path_to_store_res_df, sep='\t', header=True, index=True)


cal_post_tlc_sim()
