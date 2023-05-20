from simple_diffusion import UViT, GaussianDiffusion
from trainer import Trainer
from torchinfo import summary
import argparse

parser = argparse.ArgumentParser(description='Train diffusion model.')
parser.add_argument('--image-path', type=str, required=True, help='Path to the directory containing training images.')
parser.add_argument('--milestone', type=int, default=None, help='Load checkpoint at this milestone.')
args = parser.parse_args()

model = UViT(
    dim=16,
    dim_mults=(1, 2, 4, 8),
)

summary(model, input_size=[(1, 3, 64, 64), (1,)], verbose=1)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
)

trainer = Trainer(
    diffusion,
    args.image_path,
    train_batch_size = 128,
    train_lr = 1e-3,
    train_num_steps = 100_000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 5000,     # every 1000 steps, save checkpoint + sample imgs
    calculate_fid = True              # whether to calculate fid during training
)

if args.milestone is not None:
    trainer.load(args.milestone)

trainer.train()