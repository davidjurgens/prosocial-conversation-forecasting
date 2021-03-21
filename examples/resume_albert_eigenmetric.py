from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from pathlib import Path

ROOT = Path('/shared/0/projects/prosocial/data/finalized/')
pretrained_system_name_or_path = \
    ROOT / 'model_checkpoints/albert/run2/best_checkpoint_ep214498.pth'
input_dir = ROOT / 'data_cache/albert'
output_dir = ROOT / 'model_checkpoints/albert/run2_resume'

solver = EigenmetricRegressionSolver.from_pretrained(
    model_constructor=AlbertForEigenmetricRegression,
    pretrained_system_name_or_path=pretrained_system_name_or_path,
    resume_training=True,
    input_dir=input_dir,
    output_dir=output_dir,
    n_epoch=2,
    learning_rates=2e-6)

solver.fit(num_eval_per_epoch=10)

