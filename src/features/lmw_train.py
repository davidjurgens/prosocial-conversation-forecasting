import little_mallet_wrapper as lmw
import pandas as pd
import argparse
import os

# this line should be consistent with the header in tpcdata.py
text_col = 'text_for_topic_modeling'

def train_lmw():
    """train a topic mode
        note that the dataframe has to fit in memory
    """
    parser = argparse.ArgumentParser(description='Arguments for Eigenmetric Regression')
    parser.add_argument('--df_path', type=str, required=True, help=('path to the training data'
                        'e.g. part-00000-7c30c29e-bc47-4069-8b52-6e8ee75a429d-c000.csv'))
    parser.add_argument('--path_to_mallet', type=str, required=True,
                        help='path to the mallet module, e.g. mallet-2.0.8/bin/mallet')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='directory to store model checkpoints, e.g. models/lmw-output')
    parser.add_argument('--num_topics', default=20, type=int, required=False,
                        help='number of topics to be considered')
    args = parser.parse_args()
    # get arguments
    df = pd.read_csv(args.df_path, sep='\t', engine='python')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # basic process
    training_data = [lmw.process_string(t) for t in df[text_col].tolist()]
    training_data = [d for d in training_data if d.strip()]
    lmw.quick_train_topic_model(args.path_to_mallet, 
                                args.checkpoint_dir, 
                                args.num_topics, 
                                training_data)

train_lmw()
