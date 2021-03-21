"""Author: Yiming Zhang"""
import argparse
import logging
import torch

from utils import MODEL_CLASSES, set_seed, load_and_cache_datasets
from train import train
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--model_type", required=True, type=str, choices=list(MODEL_CLASSES.keys()), help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--sequence_length", default=128, type=int, help="Sequence length for language model."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--training_type",
        required=True,
        choices=["comment", "post"],
        help="Choose between a comment/post fine-tuned model."
    )
    parser.add_argument("--train_files", nargs='+', help='Training file(s)')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--legacy", action="store_true", help="Legacy code for compatibility with older pytorch versions.")



    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                            and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()


    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   -1, device, args.n_gpu, False, False)

    # Set seed
    set_seed(args)


 
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                cache_dir=args.cache_dir if args.cache_dir else None)


    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(
                                            '.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    dataset = load_and_cache_datasets(args, tokenizer)
    train(args, dataset, model, tokenizer)


if __name__ == '__main__':
    main()