import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from models import ALBERT_META_FEATURES, ALBERT_EIGENMETRICS
from models import NUM_SUBREDDIT_EMBEDDINGS, SUBREDDIT_EMBEDDINGS_SIZE
from runx.logx import logx


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

    save_dict = {"model_construct_params_dict": model.param_dict(),
                 "state_dict": model.state_dict()}

    logx.initialize(logdir=args.output_dir,
                    coolname=True,
                    tensorboard=False,
                    no_timestamp=False,
                    eager_flush=True)

    logx.save_model(save_dict,
                    metric=0,
                    epoch=0,
                    higher_better=False)

    # preds, golds = solver.infer(data_path=args.input_dir / 'cached.test.albert.buffer')
    # scores = solver.get_scores(preds, golds)
    # print(scores)


if __name__ == '__main__':
    main()
