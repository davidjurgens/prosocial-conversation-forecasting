from collections import OrderedDict

from runx.logx import logx
import xgboost as xgb
import pandas as pd
import joblib
import os
from pathlib import Path
import argparse
import json
import torch
from models import XGBOOST_FEATURES, EIGENMETRICS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Tuple


def neg_rsqaure_score(predt: np.ndarray, dmat: xgb.DMatrix) -> Tuple[str, float]:
    """
    negated r-sqaure score metric.
    :param predt: predicted values
    :param dmat: dmatrix
    :return: r-sqaure
    """
    gold = dmat.get_label()
    return 'NegrSquare', -r2_score(y_true=gold, y_pred=predt)


def callback_func(env):
    """
    callback function that records r^2 and MSE
    """
    if env.evaluation_result_list[0][0] == "dev-NegrSquare" and env.evaluation_result_list[1][0] == "dev-rmse":
        eval_dict = {
            "R2": -env.evaluation_result_list[0][1],
            "MSE": env.evaluation_result_list[1][1],
        }
    elif env.evaluation_result_list[0][0] == "dev-rmse" and env.evaluation_result_list[1][0] == "dev-NegrSquare":
        eval_dict = {
            "MSE": env.evaluation_result_list[0][1],
            "R2": -env.evaluation_result_list[1][1],
        }
    else:
        eval_dict = {
            env.evaluation_result_list[0][0]: env.evaluation_result_list[0][1],
            env.evaluation_result_list[1][0]: env.evaluation_result_list[1][1],
        }
    logx.metric('val',
                eval_dict,
                env.iteration)


class XgboostSolver(object):
    def __init__(self, input_dir: Path, output_dir: Path, model_construct_params_dict: OrderedDict, **kwargs):
        """
        api for xgboost
        :param args: model arguments
        :param input_dir: the path to the input directory
        :param output_dir: the path to save checkpoints
        (this parameter is also in args; add it as an explicit parameter to remind of this side effect)
        """

        self.construct_param_dict = \
            OrderedDict({
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "model_construct_params_dict": model_construct_params_dict,
            })

        # build log


        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data = kwargs.pop("data", None)
        if model_construct_params_dict['xgb_model']:
            logx.initialize(logdir=output_dir,
                            coolname=True,
                            tensorboard=False,
                            no_timestamp=False,
                            hparams={"solver_hparams": self.construct_param_dict},
                            eager_flush=True)
            logx.msg(f"loaded models from {model_construct_params_dict['xgb_model']}")
            self.bst = xgb.Booster({'nthread': 32})
            self.bst.load_model(model_construct_params_dict['xgb_model'])
            logx.msg(f"loaded pretrained model from {model_construct_params_dict['xgb_model']}")
        else:
            logx.initialize(logdir=output_dir,
                            coolname=True,
                            tensorboard=True,
                            no_timestamp=False,
                            hparams={"solver_hparams": self.construct_param_dict},
                            eager_flush=True)

    @classmethod
    def from_scratch(cls, input_dir: Path, output_dir: Path, params: dict, num_boost_round: int,
                     verbose_eval: bool, early_stopping_rounds: int):
        if output_dir.exists() and os.listdir(str(output_dir)):
            raise ValueError(f"Output directory ({output_dir}) already exists "
                             "and is not empty")
        output_dir.mkdir(exist_ok=True, parents=True)

        # get data
        if (input_dir / "train.dmatrix.pth.tar").exists() and (input_dir / "dev.dmatrix.pth.tar"):
            data = dict()
            data["train"] = xgb.DMatrix(str(input_dir / "train.dmatrix.pth.tar"))
            data["dev"] = xgb.DMatrix(str(input_dir / "dev.dmatrix.pth.tar"))
            print(f"loaded cached data from {input_dir}")
        else:
            data = dict()
            print(f"processing data from {input_dir}")
            data["train"] = cls.get_data(input_dir, "train")
            data["dev"] = cls.get_data(input_dir, "dev")
            data["train"].save_binary(str(input_dir / "train.dmatrix.pth.tar"))
            data["dev"].save_binary(str(input_dir / "dev.dmatrix.pth.tar"))
            print(f"cached data at {input_dir}")

        model_construct_params_dict = OrderedDict({
            "params": params,
            "num_boost_round": num_boost_round,
            "verbose_eval": verbose_eval,
            "early_stopping_rounds": early_stopping_rounds,
            "xgb_model": None
        })
        return cls(input_dir, output_dir, model_construct_params_dict, data=data)

    @classmethod
    def from_pretrained(cls, pretrained_system_dir: Path, input_dir: Path = None, output_dir: Path = None):

        with (pretrained_system_dir / 'xgboost.construct.json').open('r') as istream:
            solver_construct_param_dict = json.load(istream)

        con_input_dir = input_dir if input_dir else solver_construct_param_dict["input_dir"]
        con_output_dir = output_dir if output_dir else solver_construct_param_dict["output_dir"]
        model_construct_params_dict = solver_construct_param_dict["model_construct_params_dict"]
        model_construct_params_dict["xgb_model"] = str(pretrained_system_dir / "xgboost.model")

        return cls(con_input_dir, con_output_dir, model_construct_params_dict)

    @staticmethod
    def get_data(data_path: Path, mode: str) -> xgb.DMatrix:
        data = pd.read_csv(data_path / f"{mode}.tsv", sep="\t", usecols=XGBOOST_FEATURES + EIGENMETRICS)
        x = data[XGBOOST_FEATURES]
        y = data[EIGENMETRICS]
        dmatrix = xgb.DMatrix(data=x, label=y)
        return dmatrix

    def fit(self, test_path: Path = None):
        evals = [(self.data["dev"], "dev")]
        train_args = {
            "dtrain": self.data["train"],
            "evals": evals,
            "feval": neg_rsqaure_score,
            **self.construct_param_dict["model_construct_params_dict"],
            "callbacks": [callback_func],
        }

        bst = xgb.train(**train_args)
        logx.msg(f"best score: {bst.best_score}, best iteration: {bst.best_ntree_limit}")

        self.bst = bst
        self.save_checkpoints()
        logx.msg(f"Saved model at {self.output_dir}")

        if test_path:
            self.data = None  # no longer need the big dataset
            if (test_path.parents[0] / 'test.dmatrix.x.pth.tar').exists() and \
                    (test_path.parents[0] / 'test.dmatrix.y.pth.tar').exists():
                dtest = xgb.DMatrix(str(test_path.parents[0] / 'test.dmatrix.x.pth.tar'))
                ygold = torch.load(test_path.parents[0] / 'test.dmatrix.y.pth.tar')
            else:
                final_eval_data = pd.read_csv(test_path, sep="\t", usecols=XGBOOST_FEATURES + EIGENMETRICS)
                dtest = xgb.DMatrix(final_eval_data[XGBOOST_FEATURES].values)
                ygold = final_eval_data[EIGENMETRICS].values
                # cache test data
                dtest.save_binary(str(self.input_dir / "test.dmatrix.x.pth.tar"))
                torch.save(ygold, self.input_dir / "test.dmatrix.y.pth.tar")
                print(f"cached test data at {self.input_dir}")

            ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
            logx.msg(f'rSqaure on test set: {r2_score(ygold, ypred)}')
            logx.msg(f'MSE on test set: {mean_squared_error(ygold, ypred)}')
            output_path = self.output_dir / 'final_eval_pred.pth.tar'
            joblib.dump({'pred': ypred, 'gold': ygold}, output_path)
            logx.msg(f'Stored eval results in {output_path}')

    def save_checkpoints(self):
        with (self.output_dir / 'xgboost.construct.json').open('w') as ostream:
            ostream.write(json.dumps(self.construct_param_dict))
        self.bst.save_model(str(self.output_dir / "xgboost.model"))

    def infer(self, dmatx: xgb.DMatrix):
        num_parallel_tree = 1  # handcoded
        best_ntree_limit = (int(self.bst.attributes()['best_iteration']) + 1) * num_parallel_tree
        return self.bst.predict(dmatx, ntree_limit=best_ntree_limit)

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser(description='train xgboost models')
        parser.add_argument('--input_dir', type=Path, default='input',
                            help='The directory of the input files.')
        parser.add_argument('--output_dir', type=Path, default='output',
                            help=' The directory to store models and arguments.')
        # general parameters
        parser.add_argument('--verbose_eval', type=int, default=2,
                            help='Verbosity of printing messages. '
                                 'Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).')
        parser.add_argument('--num_boost_round', type=int, default=3000,
                            help='The number of rounds for boosting')
        parser.add_argument('--booster', type=str, default='gbtree',
                            help='Can be gbtree, gblinear or dart; '
                                 'gbtree and dart use tree based models '
                                 'while gblinear uses linear functions.')
        # booster parameters
        parser.add_argument('--eta', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Minimum loss reduction required to make a '
                                 'further partition on a leaf node of the tree.')
        parser.add_argument('--max_depth', type=int, default=4,
                            help='Maximum depth of a tree. Increasing this value will '
                                 'make the model more complex and more likely to overfit.')
        parser.add_argument('--min_child_weight', type=float, default=1,
                            help='Minimum sum of instance weight (hessian) needed in a child.')
        parser.add_argument('--subsample', default=0.8, type=float,
                            help='Subsample ratio of the training instances. Setting it to 0.5 means '
                                 'that XGBoost would randomly sample half of the training data prior '
                                 'to growing trees. ')
        parser.add_argument('--colsample_bytree', type=float, default=0.8,
                            help='the subsample ratio of columns when constructing each tree')
        parser.add_argument('--l2reg', type=float, default=3,
                            help='L2 regularization term on weights. '
                                 'Increasing this value will make model more conservative.')
        parser.add_argument('--l1reg', type=float, default=1,
                            help='L1 regularization term on weights. '
                                 'Increasing this value will make model more conservative.')
        # parser.add_argument('--tree_method', type=str, default='auto',
        #                     help='Choices: auto, exact, approx, hist, gpu_hist, '
        #                          'this is a combination of commonly used updaters.')
        # parser.add_argument('--predictor', type=str, default='cpu_predictor',
        #                     help='The type of predictor algorithm to use. '
        #                          'Provides the same results but allows the use of GPU or CPU.')
        parser.add_argument('--num_parallel_tree', type=int, default=1,
                            help='Number of parallel trees constructed during each iteration')
        args = parser.parse_args()

        params = {'verbose_eval': args.verbose_eval, 'eta': args.eta, 'gamma': args.gamma,
                  'max_depth': args.max_depth, 'min_child_weight': args.min_child_weight,
                  'subsample': args.subsample, 'colsample_bytree': args.colsample_bytree,
                  'lambda': args.l2reg, 'alpha': args.l1reg, 'num_parallel_tree': args.num_parallel_tree,
                  'num_boost_round': args.num_boost_round}  # remove 'tree_method': args.tree_method, 'predictor': args.predictor

        return params, args
