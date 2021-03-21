from __future__ import absolute_import, division, print_function

import os
import logging
import random

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, Subset)
from tqdm.auto import tqdm


from tqdm import tqdm

from sklearn.metrics import r2_score


""" from learn_with_meta_data.evaluate import (evaluate, quick_evaluate) """

logger = logging.getLogger(__name__)

def simple_mse(preds, labels):
    return np.mean((np.array(preds - labels)) ** 2)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return simple_mse(preds, labels), r2_score(labels, preds)



def evaluate(args, model, eval_dataset, tb_writer):
    """ Evaluate the model """
    batch_size = args.batch_size if args.device == torch.device('cpu') else args.batch_size * args.n_gpu
    eval_dataloader = DataLoader(eval_dataset, shuffle=False,
                            batch_size=batch_size)
    logging.info("***** Running evaluation *****")
    logging.info(f"  Num examples = {len(eval_dataset)}")

    preds = []
    labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            batch = [t.to(args.device) for t in batch]
            pc0 = batch[-1]
            loss, logits = model(batch)
            logits = logits.squeeze().cpu().numpy()
            pc0 = pc0.squeeze().cpu().numpy()
            if logits.ndim == 0:
                preds.append(logits.tolist())
                labels.append(pc0.tolist())
            else:
                preds.extend(logits.tolist())
                labels.extend(pc0.tolist())

            

    preds = np.array(preds).reshape(-1)
    labels = np.array(labels).reshape(-1)
    mse, r2 = compute_metrics(preds,labels)
    tb_writer.add_scalar('mse_test_set', mse, 0)
    tb_writer.add_scalar('r2_test_set', r2, 0)
    logging.info("MSE: {}".format(mse))
    logging.info("R2: {}".format(r2))

def quick_evaluate(args, model, eval_dataset, tb_writer, step, ratio=1000):
    """ Evaluate the model """
    batch_size = args.batch_size if args.device == torch.device('cpu') else args.batch_size * args.n_gpu
    subset_size = max(len(eval_dataset) // ratio, 2)
    eval_dataset = Subset(eval_dataset, random.sample(range(len(eval_dataset)), subset_size))
    eval_dataloader = DataLoader(eval_dataset, shuffle=False,
                            batch_size=batch_size)
    logging.info("***** Running quick evaluation @ step {} *****".format(step))
    logging.info(f"  Num examples = {len(eval_dataset)}")
    preds = []
    labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
                batch = [t.to(args.device) for t in batch]
                pc0 = batch[-1]
                loss, logits = model(batch)
                logits = logits.squeeze().cpu().numpy()
                pc0 = pc0.squeeze().cpu().numpy()
                if logits.ndim == 0:
                    preds.append(logits.tolist())
                    labels.append(pc0.tolist())
                else:
                    preds.extend(logits.tolist())
                    labels.extend(pc0.tolist())



    preds = np.array(preds).reshape(-1)
    labels = np.array(labels).reshape(-1)
    mse, r2 = compute_metrics(preds,labels)
    tb_writer.add_scalar('mse_quick_eval', mse, step)
    tb_writer.add_scalar('r2_quick_eval', r2, step)
    logging.info("MSE: {}".format(mse))
    logging.info("R2: {}".format(r2))

    return r2
    
    
                
    




