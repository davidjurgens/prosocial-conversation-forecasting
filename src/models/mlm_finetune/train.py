"""Author: Yiming Zhang"""
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler
from utils import *



logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter(os.path.join(args.output_dir, args.training_type, 'tb'))
    """ Train the model """
    tb_writer = SummaryWriter()
    args.train_batch_size = 2 if args.device=='cpu' else args.per_gpu_train_batch_size * max(1, args.n_gpu) 
    train_sampler = RandomSampler(
        train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    # Make mask token backward compatible
    if args.legacy:
        _mask_tokens = mask_tokens_legacy
    else:
        _mask_tokens = mask_tokens
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Dataset Size = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    model.train()

    train_iterator = tqdm(range(int(args.num_train_epochs)),
                            desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_loss = 0.0
        logging.info("   Begin Epoch {}   ".format(epoch))
        logging_numbers = [0.0]
        for step, (tokens, mask) in enumerate(epoch_iterator):
            try:
                inputs, labels = _mask_tokens(
                    tokens, mask, tokenizer, args)
            except TypeError:
                _mask_tokens = mask_tokens_legacy
                inputs, labels = _mask_tokens(
                    tokens, mask, tokenizer, args)
            tokens = tokens.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            mask = mask.to(args.device)
            

            outputs = model(inputs, attention_mask=mask, masked_lm_labels=labels)
            # Get masked-lm loss & last hidden state
            lm_loss = outputs[0]
            
            if args.n_gpu > 1:
                lm_loss = lm_loss.mean()

            lm_loss.backward()
            optimizer.step()
            model.zero_grad()
            epoch_loss += lm_loss.item()

            global_step += 1

            logging_numbers[0] += lm_loss.item()

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not average well
                logging.info('Step: {}'.format(global_step))
                logging.info('MLM Loss: {}'.format(logging_numbers[0]/args.logging_steps))

                tb_writer.add_scalar('MLM_Loss', logging_numbers[0]/args.logging_steps, global_step)
                logging_numbers = [0.0]

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, args.training_type, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(
                    output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        del tokens, labels, mask, inputs
        tb_writer.add_scalar('epoch_loss', epoch_loss / (step+1), epoch)
        logging.info("Epoch average loss: {}".format(epoch_loss / (step+1)))
        

        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    
    tb_writer.close()

    return global_step, tr_loss / global_step

