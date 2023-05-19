from simple_diffusion import UViT, GaussianDiffusion
from trainer import Trainer

model = UViT(
    dim=64
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 1e-3,
    train_num_steps = 100_000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 5000,     # every 1000 steps, save checkpoint + sample imgs
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()