from models.xgboost.XgboostSolver import XgboostSolver
import joblib


def main():
    params, args = XgboostSolver.get_arguments()
    solver = XgboostSolver.from_scratch(input_dir=args.input_dir,
                                        output_dir=args.output_dir,
                                        params=params,
                                        num_boost_round=args.num_boost_round,
                                        verbose_eval=args.verbose_eval,
                                        early_stopping_rounds=50)
    solver.fit(test_path=args.input_dir / 'test.tsv')
    joblib.dump(solver.bst, args.output_dir / 'bst.pth.tar')


if __name__ == '__main__':
    main()
