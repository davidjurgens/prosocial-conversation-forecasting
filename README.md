Measuring and Forecasting Prosocial Behavior in Conversations
==============================

This repository contains code and models for the reproducing the paper "Conversations Gone Alright: Quantifying and Predicting Prosocial Outcomes in Online Conversations" at the Web Conference 2021

Full documentations are forthcoming.

Quick tour
------------
prosocial-conversation-forecasting is a tool that forecasts prosocial conversation outcomes from early conversation signals.
It is released along with our TheWebConf2021 paper [Conversations Gone Alright: Quantifying and Predicting
Prosocial Outcomes in Online Conversations](https://arxiv.org/pdf/2102.08368.pdf) for details. This repo contains all code for preprocessing, evaluating and modeling.

### Installation:
- From source
```shell
git clone https://github.com/davidjurgens/prosocial-conversation-forecasting.git
cd prosocial-conversation-forecasting
pip install -r requirements.txt
```

### Process Dataset from scratch:
- get basic features on spark
```shell
 spark-submit 
   --master yarn 
   --queue default --conf spark.ui.port=6006 
   --conf spark.executorEnv.PYTHONPATH=path/to/site-packages 
   --executor-memory 16g 
   --num-executors 8 
   src/features/data.py 
   --input_file path/to/raw/jsonl/data  
   --output_file path/to/store/results 
   --num_executors 8
```

- preprocess for topic modeling features:
```shell
spark-submit 
   --master yarn 
   --queue workshop --conf spark.ui.port=6006 
   --conf spark.executorEnv.PYTHONPATH=path/to/site-packages 
   --executor-memory 16g 
   --num-executors 8 
   src/features/tpcdata.py 
   --input_file path/to/results/from/last/step  
   --output_file research-out/text_for_topic_modeling 
   --stop_words_file path/to/store/results 
   --low_freq_threshold 10 
   --num_executors 8

```

- extract topic features
```shell
# train a topic model
python src/features/lmw_train.py 
       --df_path /shared/0/projects/prosocial/part-00000-7c30c29e-bc47-4069-8b52-6e8ee75a429d-c000.csv 
       --checkpoint_dir models/lmw-output 
       --path_to_mallet /home/jiajunb/mallet-2.0.8/bin/mallet
# infer with the model
python src/features/lmw_infer.py 
       --infer_df_path /shared/0/projects/prosocial/train.newimp.tsv 
       --split train 
       --path_to_mallet /home/jiajunb/mallet-2.0.8/bin/mallet 
       --path_to_model models/lmw-output/mallet.model.20 
       --tmp_dir data/interim 
       --path_to_store_res_df /shared/0/projects/prosocial/finalized/train.tsv
```

### Train and infer the models:
Examples for training and inferring models are available [here](examples)

You can download our model checkpoints [here](https://drive.google.com/file/d/1_o5h6ChNmwRygPXDJ2imUOyOqlgAsPf8/view?usp=sharing), and our processed datasets [here](https://drive.google.com/drive/folders/1f4Nq643htIEaRPQDu2X0U5e2QbxNZdAx?usp=sharing)

- train an albert-based model to predict prosocial trajectory
```shell
export PRETRAINED_MODELS_ROOT_DIR=path/to/pretrained/model
export DATASET_ROOT_DIR=path/to/the/dataset

# refer to the appendix of our paper for the actual hyperparameters
/opt/anaconda/bin/python examples/run_albert_eigenmetric.py 
    --top_comment_pretrained_model_name_or_path albert-base-v2 
    --post_pretrained_model_name_or_path albert-base-v2 
    --classifier_dropout_prob 0.5 
    --subreddit_pretrained_path ${PRETRAINED_MODELS_ROOT_DIR}/subreddit_embeddings/pretrained_subreddit_embeddings.tar.pth 
    --input_dir ${DATASET_ROOT_DIR}/data_cache/albert 
    --output_dir ${DATASET_ROOT_DIR}/model_checkpoints/albert/run1_freeze 
    --learning_rate 1e-4 
    --num_eval_per_epoch 10
    --n_epoch 5 
    --per_gpu_batch_size 890 
    --freeze_alberts 
    --weight_decay 1e-6
```

- train a model to rank conversations

```shell
/opt/anaconda/bin/python examples/run_annotation_albert.py 
    --pretrained_feature_extractor_name_or_path path/to/the/pretrained/model 
    --freeze_feature_extractor 
    --input_dir path/to/the/training/data
    --output_dir path/to/store/checkpoints 
    --learning_rate 1e-4 
    --num_eval_per_epoch 1 
    --n_epoch 200 
    --per_gpu_batch_size 800 
    --weight_decay 1e-6
```


Project Organization
------------
```
.
├── LICENSE
├── Makefile
├── README.md
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── models
│   ├── english
│   └── english.pickle
├── notebooks
│   ├── 0-SVD-on-training-set.ipynb
│   ├── 1-extract-principle-components.ipynb
│   └── 2-pretrained-subreddit-embeddings-processing.ipynb
├── references
├── reports
│   ├── dynamic_analysis
│   │   └── spark-src
│   │       ├── 0_downsample_data.py
│   │       ├── 1_extract_basic_metrics.py
│   │       └── 2_merge_metric_dfs.py
│   └── figures
├── requirements.txt
├── setup.py
├── src
│   ├── features
│   │   ├── data.py
│   │   ├── filter.py
│   │   ├── junjie_features.py
│   │   ├── lmw_infer.py
│   │   ├── lmw_train.py
│   │   └── tpcdata.py
│   └── models
│       ├── albert
│       │   ├── AlbertForEigenmetricRegression.py
│       │   ├── CommentsDataset.py
│       │   └── EigenmetricRegressionSolver.py
│       ├── albert_for_linreg
│       │   ├── eval.py
│       │   ├── main.py
│       │   ├── model.py
│       │   ├── train.py
│       │   └── utils.py
│       ├── annotation_albert
│       │   ├── AlbertForProsocialnessSolver.py
│       │   └── BaselineDataset.py
│       ├── bases
│       │   └─── Solver.py
│       │   
│       ├── fusedModel
│       │   ├── FusedDataset.py
│       │   ├── FusedModelSolver.py
│       │   └── FusedPredictor.py
│       ├── mlm_finetune
│       │   ├── main.py
│       │   ├── run.sh
│       │   ├── run_post.sh
│       │   ├── train.py
│       │   └── utils.py
│       ├── predict_model.py
│       ├── train_model.py
│       └── xgboost
│           └── XgboostSolver.py
├── test_environment.py
└── tox.ini

18 directories, 59 files
```

