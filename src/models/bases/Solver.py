"""A Solver for Single-node servers"""
import os
import random
from pathlib import Path
from collections import OrderedDict
from typing import Tuple
import argparse

import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import numpy
from runx.logx import logx
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import r2_score


class BasicSolver(object):

    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 model: nn.Module,
                 device: torch.device,
                 per_gpu_batch_size: int,
                 n_gpu: int,
                 batch_size: int,
                 learning_rate: float,
                 weight_decay: float,
                 n_epoch: int,
                 seed: int,
                 **kwargs):
        # construct param dict
        self.construct_param_dict = OrderedDict({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "learning_rate": learning_rate,
            "n_epoch": n_epoch,
            "per_gpu_batch_size": per_gpu_batch_size,
            "weight_decay": weight_decay,
            "seed": seed
        })

        self.subreddit_embedding_device = torch.device("cuda")

        # build log
        logx.initialize(logdir=output_dir,
                        coolname=True,
                        tensorboard=True,
                        no_timestamp=False,
                        hparams={"solver_hparams": self.state_dict(), "model_hparams": model.param_dict()},
                        eager_flush=True)
        # arguments
        self.input_dir = input_dir
        self.output_dir = output_dir

        # training utilities
        self.model = model

        # data utilities
        self.train_dataloader = kwargs.pop("train_dataloader", None)
        self.dev_dataloader = kwargs.pop("dev_dataloader", None)
        self.batch_size = batch_size

        self.n_epoch = n_epoch
        self.seed = seed
        # device
        self.device = device
        self.n_gpu = n_gpu
        logx.msg(f'Number of GPU: {self.n_gpu}.')

        self.criterion = nn.MSELoss()

        # optimizer and scheduler
        if self.train_dataloader:
            self.optimizer, self.scheduler = self.get_optimizer(named_parameters=self.model.named_parameters(),
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                train_dataloader=self.train_dataloader,
                                                                n_epoch=n_epoch)
        # set up random seeds and model location
        self.setup()

    @classmethod
    def from_scratch(cls,
                     model: nn.Module,
                     input_dir: Path,
                     output_dir: Path,
                     learning_rate: float,
                     n_epoch: int,
                     per_gpu_batch_size: int,
                     weight_decay: float,
                     seed: int):
        # check the validity of the directory
        if (output_dir).exists() and os.listdir(str(output_dir)):
            raise ValueError(f"Output directory ({output_dir}) already exists "
                             "and is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        # data utilities
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        n_gpu = torch.cuda.device_count()
        batch_size = per_gpu_batch_size * max(1, n_gpu)

        # dataloader
        train_dataloader = cls.get_train_dataloader(input_dir, batch_size)
        dev_dataloader = cls.get_dev_dataloader(input_dir, batch_size)

        return cls(input_dir, output_dir, model, device, per_gpu_batch_size, n_gpu, batch_size, learning_rate,
                   weight_decay, n_epoch, seed, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)

    @classmethod
    def from_pretrained(cls,
                        model_constructor: nn.Module,
                        pretrained_system_name_or_path: Path,
                        resume_training: bool = False,
                        input_dir: Path = None,
                        output_dir: Path = None,
                        n_epoch: int = None,
                        **kwargs):
        # load checkpoints
        checkpoint = torch.load(pretrained_system_name_or_path)
        state_dict = checkpoint['state_dict']
        meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}

        # load model
        model_key = kwargs.pop("model_key", "model_construct_params_dict")
        model = model_constructor(**meta[model_key])
        model.load_state_dict(state_dict)
        model.eval()

        # load arguments
        solver_args = meta["solver_construct_params_dict"]
        solver_args["model"] = model
        # update some parameters
        solver_args["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        solver_args["n_gpu"] = torch.cuda.device_count()
        # load dataset
        if resume_training:
            if input_dir is None or output_dir is None or n_epoch is None:
                raise AssertionError("Either input_dir, output_dir or n_epoch (for resuming) is None!")

            solver_args["n_epoch"] = n_epoch
            solver_args["input_dir"] = input_dir
            solver_args["output_dir"] = output_dir
            solver_args["weight_decay"] = kwargs.pop("weight_decay", solver_args["weight_decay"])
            solver_args["learning_rates"] = kwargs.pop("learning_rate", solver_args["learning_rates"])
            solver_args["per_gpu_batch_size"] = kwargs.pop("per_gpu_batch_size", solver_args["per_gpu_batch_size"])
            solver_args["train_dataloader"] = cls.get_train_dataloader(input_dir, solver_args["batch_size"])
            solver_args["dev_dataloader"] = cls.get_dev_dataloader(input_dir, solver_args["batch_size"])
        # calculate
        solver_args["batch_size"] = solver_args["per_gpu_batch_size"] * max(1, solver_args["n_gpu"])
        return cls(**solver_args)

    def state_dict(self) -> OrderedDict:
        return OrderedDict(self.construct_param_dict)

    def fit(self, num_eval_per_epoch: int = 10):
        steps_per_eval = len(self.train_dataloader) // num_eval_per_epoch
        self.train(steps_per_eval)
        test_dataloader = self.get_test_dataloader(self.input_dir, self.batch_size)
        mean_loss, metrics_scores, _, _ = self.validate(test_dataloader)
        logx.msg("Scores on test set: ")
        logx.msg(str(metrics_scores))

    def setup(self):
        # put onto cuda
        self.model = self.model.to_device(main_device=self.device,
                                          embedding_device=self.subreddit_embedding_device,
                                          data_parallel=(self.n_gpu > 1))
        # fix random seed
        self.fix_random_seed()

    def fix_random_seed(self):
        # Set seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def train(self, steps_per_eval: int):
        # TensorBoard
        for epoch_idx in range(self.n_epoch):
            self.__train_per_epoch(epoch_idx, steps_per_eval)

    def validate(self, dataloader: DataLoader) -> Tuple[torch.tensor, dict, torch.tensor, torch.tensor]:
        preds, golds = self.__forward_batch_plus(dataloader)
        preds = preds.detach().cpu()
        golds = golds.detach().cpu()
        mean_loss = self.criterion(preds.view(golds.shape), golds)  # num_of_label should be 1 for it to work
        if self.n_gpu > 1:
            mean_loss = mean_loss.mean()  # mean() to average on multi-gpu.
        metrics_scores = self.get_scores(preds.view(golds.shape), golds)
        return mean_loss, metrics_scores, preds, golds

    def __train_per_epoch(self, epoch_idx: int, steps_per_eval: int):
        with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch_idx}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                global_step = epoch_idx * len(self.train_dataloader) + batch_idx
                loss = self.__training_step(batch)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()
                logx.metric('train', {"tr_loss": loss.item(),
                                      "learning_rate": self.scheduler.get_last_lr()[0]}, global_step)
                pbar.set_postfix_str(f"tr_loss: {loss.item():.5f}")
                # update weights
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                if batch_idx % steps_per_eval == 0:
                    # validate and save checkpoints
                    # downsample a subset of dev dataset
                    eval_dataset = self.dev_dataloader.dataset
                    subset_size = len(eval_dataset) // 500
                    eval_sampled_dataloader = DataLoader(
                        Subset(self.dev_dataloader.dataset, random.sample(range(len(eval_dataset)), subset_size)),
                        shuffle=True,
                        batch_size=self.batch_size,
                        pin_memory=True)
                    mean_loss, metrics_scores, _, _ = self.validate(eval_sampled_dataloader)
                    logx.metric('val', metrics_scores, global_step)
                    if self.n_gpu > 1:
                        save_dict = {"model_construct_params_dict": self.model.module.param_dict(),
                                     "state_dict": self.model.module.state_dict(),
                                     "solver_construct_params_dict": self.state_dict(),
                                     "optimizer": self.optimizer.state_dict()}
                    else:
                        save_dict = {"model_construct_params_dict": self.model.param_dict(),
                                     "state_dict": self.model.state_dict(),
                                     "solver_construct_params_dict": self.state_dict(),
                                     "optimizer": self.optimizer.state_dict()}

                    logx.save_model(save_dict,
                                    metric=mean_loss,
                                    epoch=global_step,
                                    higher_better=False)
                pbar.update(1)

    def batch_to_device(self, batch) -> Tuple[dict, torch.tensor]:
        """
        example:

        :param batch:
        :return:

        eta_data, subreddit_ids, input_ids_tlc, attention_mask_tlc, \
            input_ids_post, attention_mask_post, labels = batch
        batch_input = {
            'meta_data': meta_data.to(self.device),
            'subreddit_ids': subreddit_ids.to(torch.int64).to(self.subreddit_embedding_device),
            'input_ids_tlc': input_ids_tlc.to(torch.int64).to(self.device),
            'attention_mask_tlc': attention_mask_tlc.to(torch.int64).to(self.device),
            'input_ids_post': input_ids_post.to(torch.int64).to(self.device),
            'attention_mask_post': attention_mask_post.to(torch.int64).to(self.device),
            'labels': labels.to(self.device)
        }
        return batch_input, batch_input["labels"]
        """
        pass

    def __training_step(self, batch) -> torch.tensor:
        """
        a single forwarding step for training

        :param self: a solver
        :param batch: a batch of input for model
        :return: training loss for this batch
        """
        self.model.zero_grad()  # reset gradient
        self.model.train()
        batch_input, _ = self.batch_to_device(batch)
        outputs = self.model(**batch_input)
        loss = outputs[0]
        return loss

    def __forwarding_step(self, batch):
        """
        a single forwarding pass for the validation
        e.g.
        meta_features, input_ids, input_mask, segment_ids, labels = batch
        batch_input = {'meta_features': meta_features.to(self.device),
                       'input_ids': input_ids.to(self.device),
                       'attention_mask': input_mask.to(self.device),
                       'token_type_ids': segment_ids.to(self.device),
                       'labels': labels}
        logits = self.model(**batch_input)
        return logits, labels

        :param self: a solver
        :param batch: a batch of input for model
        :return: logits and ground true label for this batch
        """
        self.model.eval()
        batch_input, labels = self.batch_to_device(batch)
        outputs = self.model(**batch_input)
        logits = outputs[1]
        return logits.cpu().detach(), labels.cpu().detach()

    @staticmethod
    def get_scores(preds, golds) -> dict:
        """
        It is going to be registered.
        :param preds:
        :param golds:
        :return: a dictionary of all the measure
        """
        loss_measure = nn.MSELoss()
        return {'R2': r2_score(y_true=golds, y_pred=preds),
                'MSE': loss_measure.forward(input=preds, target=golds).item()}

    def __forward_batch_plus(self, dataloader: DataLoader):
        preds_list = list()
        golds_list = list()
        with tqdm(total=len(dataloader), desc=f"Evaluating: ") as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    logits, labels = self.__forwarding_step(batch)
                    preds_list.append(logits)
                    golds_list.append(labels)
                    pbar.update(1)
        # collect the whole chunk
        preds = torch.cat(preds_list, dim=0).cpu()
        golds = torch.cat(golds_list, dim=0).cpu()
        return preds, golds

    @classmethod
    def get_train_dataloader(cls, input_dir: Path, batch_size: int) -> DataLoader:
        encoded_data_path = input_dir / 'cached.train.albert.buffer'
        dataset = cls.__set_dataset(encoded_data_path)
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
        return train_dataloader

    @classmethod
    def get_dev_dataloader(cls, input_dir: Path, batch_size: int) -> DataLoader:
        encoded_data_path = input_dir / 'cached.dev.albert.buffer'
        dataset = cls.__set_dataset(encoded_data_path)
        dev_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
        return dev_dataloader

    @classmethod
    def get_test_dataloader(cls, input_dir: Path, batch_size: int) -> DataLoader:
        encoded_data_path = input_dir / 'cached.test.albert.buffer'
        dataset = cls.__set_dataset(encoded_data_path)
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
        return test_dataloader

    @classmethod
    def __set_dataset(cls, encoded_data_path: Path):
        return torch.load(encoded_data_path)

    def infer(self, data_path: Path):
        data_path = Path(data_path)
        dataset = self.__set_dataset(data_path)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)
        preds, golds = self.__forward_batch_plus(dataloader)
        return preds, golds

    @staticmethod
    def get_optimizer(named_parameters, learning_rate, weight_decay, train_dataloader, n_epoch):
        """
        get the optimizer and the learning rate scheduler
        :param named_parameters:
        :param learning_rate:
        :param weight_decay:
        :param train_dataloader:
        :param n_epoch:
        :return:
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_parameters if not any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
            {'params': [p for n, p in named_parameters if any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # get a linear scheduler
        num_steps_epoch = len(train_dataloader)
        # ReduceLROnPlateau(self.optimizer, 'min')
        num_train_optimization_steps = int(num_steps_epoch * n_epoch) + 1
        warmup_steps = 100
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        return optimizer, scheduler

    @staticmethod
    def get_eigenmetric_regression_arguments():
        pass

