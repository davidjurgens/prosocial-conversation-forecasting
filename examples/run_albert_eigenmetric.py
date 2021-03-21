import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from models import ALBERT_META_FEATURES, ALBERT_EIGENMETRICS
from models import NUM_SUBREDDIT_EMBEDDINGS, SUBREDDIT_EMBEDDINGS_SIZE


def main():
    args = EigenmetricRegressionSolver.get_eigenmetric_regression_arguments()
    model = AlbertForEigenmetricRegression.from_scratch(
        num_labels=len(ALBERT_EIGENMETRICS),
        top_comment_pretrained_model_name_or_path=args.top_comment_pretrained_model_name_or_path,
        post_pretrained_model_name_or_path=args.post_pretrained_model_name_or_path,
        classifier_dropout_prob=args.classifier_dropout_prob,
        meta_data_size=len(ALBERT_META_FEATURES),
        subreddit_pretrained_path=args.subreddit_pretrained_path,
        num_subreddit_embeddings=NUM_SUBREDDIT_EMBEDDINGS,
        subreddit_embeddings_size=SUBREDDIT_EMBEDDINGS_SIZE
    )
    if args.freeze_alberts:
        model = model.freeze_bert()
    solver = EigenmetricRegressionSolver.from_scratch(
        model,
        args.input_dir,
        args.output_dir,
        args.learning_rate,
        args.n_epoch,
        args.per_gpu_batch_size,
        args.weight_decay,
        args.seed
    )
    solver.fit(num_eval_per_epoch=args.num_eval_per_epoch)
    # preds, golds = solver.infer(data_path=args.input_dir / 'cached.test.albert.buffer')
    # scores = solver.get_scores(preds, golds)
    # print(scores)


if __name__ == '__main__':
    main()


