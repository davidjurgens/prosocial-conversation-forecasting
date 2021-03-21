from __future__ import absolute_import, division, print_function

import os
import logging
from torch.utils.data import (DataLoader, RandomSampler)
from tqdm.auto import tqdm
import torch
from copy import deepcopy
from src.models.albert_for_linreg.eval import quick_evaluate


""" from learn_with_meta_data.evaluate import (evaluate, quick_evaluate) """

logger = logging.getLogger(__name__)



def train(args, model, train_dataset, eval_dataset, optimizer, tb_writer):
    """ Train the model """

 

    best_model = None
    best_r2 = float('-inf')
    batch_size = args.batch_size if args.device == torch.device('cpu') else args.batch_size * args.n_gpu
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                            batch_size=batch_size)
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    model.zero_grad()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_dataloader:
                batch = [t.to(args.device) for t in batch]
                loss, logits = model(batch)
                
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()
                pbar.update(1)

                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("training_loss", loss.item(), global_step)
                    pbar.set_postfix_str(f"training_loss: {loss.item():.5f}")
                if args.evaluate_during_training_steps > 0 and \
                     global_step % args.evaluate_during_training_steps == 0:
                    
                    r2 = quick_evaluate(args, model, eval_dataset, tb_writer, global_step)
                    if r2 > best_r2:
                        best_r2 = r2
                        model_to_save = model.module if hasattr(model, 'module') else model
                        best_model = deepcopy(model_to_save)
                    model.train()
                global_step += 1
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dir = os.path.join(args.output_dir, 'Epoch_' + str(epoch))
        os.makedirs(save_dir)
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'model.bin'))
        logger.info("Saving model checkpoint to %s", save_dir)

    return best_model if best_model else model



