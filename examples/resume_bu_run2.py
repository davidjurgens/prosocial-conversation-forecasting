from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from pathlib import Path

ROOT = Path('/shared/0/projects/prosocial/data/finalized/')
pretrained_system_name_or_path = \
    ROOT / 'model_checkpoints/albert/bu_run3/last_checkpoint_ep268114.pth'
input_dir = ROOT / 'data_cache/albert'
output_dir = ROOT / 'model_checkpoints/albert/bu_run3_resume'

solver = EigenmetricRegressionSolver.from_pretrained(
    model_constructor=AlbertForEigenmetricRegression,
    pretrained_system_name_or_path=pretrained_system_name_or_path,
    resume_training=True,
    input_dir=input_dir,
    output_dir=output_dir,
    n_epoch=2,
    learning_rate=5e-6)

solver.fit(num_eval_per_epoch=10)

