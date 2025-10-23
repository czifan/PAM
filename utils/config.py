import argparse

def build_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp', type=str, default="PropNet")
    
    parser.add_argument('--model', type=str, default="PropNet")
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--n_stages', type=int, default=6, help='Number of stages')
    parser.add_argument('--deep_supervision', type=bool, default=True, help='Use deep supervision')
    parser.add_argument('--n_attn_stage', type=int, default=4, help='Stage at which attention is applied')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--conv_dim', type=int, default=2)
    parser.add_argument('--max_channels', type=int, default=512)

    parser.add_argument('--img_encoder_embed_dim', type=int, default=768)
    parser.add_argument('--img_encoder_depth', type=int, default=12)
    parser.add_argument('--img_encoder_num_heads', type=int, default=12)
    parser.add_argument("--img_encoder_global_attn_indexes", nargs='+', type=int, default=[2, 5, 8, 11])
    parser.add_argument('--mask_encoder_embed_dim', type=int, default=256)
    parser.add_argument('--mask_encoder_depth', type=int, default=6)
    parser.add_argument('--mask_encoder_num_heads', type=int, default=8)
    parser.add_argument("--mask_encoder_global_attn_indexes", nargs='+', type=int, default=[2, 5, 8, 11])
    parser.add_argument('--vit_patch_size', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--prompt_embed_dim', type=int, default=256)
    parser.add_argument('--from_scratch_ratio', type=float, default=0.2)

    parser.add_argument('--ema_warmup_epochs', type=int, default=3)
    parser.add_argument('--ema_decay', type=float, default=0.95)

    parser.add_argument('--criterion', type=str, default='CombinedCriterion')
    parser.add_argument("--criterion_name_lst", nargs='+', type=str, default=['MultiScaleSoftDiceLoss',])
    parser.add_argument("--criterion_weight_lst", nargs='+', type=float, default=[1.0,])
    parser.add_argument('--focalloss_alpha', type=float, default=0.25)
    parser.add_argument('--focalloss_gamma', type=float, default=2.0)

    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ExponentialLR'])
    parser.add_argument('--T_max', type=int, default=20)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    parser.add_argument('--dataset', type=str, default="PropDataset")
    parser.add_argument("--eval_datasets", nargs='+', type=str, default=[])
    parser.add_argument('--eval_split_dir', type=str, default="data/task_split_data")
    parser.add_argument('--data_dir', type=str, default="data/task_data_nsz20")
    parser.add_argument('--train_file', type=str, default="data/split_data/s0/train.txt")
    parser.add_argument('--val_file', type=str, default="data/split_data/s0/valid.txt")
    parser.add_argument('--eval_file', type=str, default="data/split_data/s0/valid.txt")
    parser.add_argument('--sampling_N_queries', type=int, default=4)
    parser.add_argument('--center_crop_size', type=int, default=256)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--train_iter_samples', type=int, default=-1)
    parser.add_argument('--eval_iter_samples', type=int, default=-1)
    parser.add_argument('--save_model_per_epochs', type=int, default=10)
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--display', type=int, default=32)
    parser.add_argument('--focus_metric', type=str, default="proposing_region_dice")
    
    parser.add_argument('--save_dir', type=str, default="results")
    parser.add_argument('--checkpoint', type=str, default="None")
    parser.add_argument('--checkpoint_prefix', type=str, default="module.")
    
    args = parser.parse_args()

    args.cumulative_grad_batchs = args.batch_size // args.train_batch_size
    
    return args