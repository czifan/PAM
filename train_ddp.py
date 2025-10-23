from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
from torch.cuda.amp import GradScaler, autocast
from imports import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    
def train_one_epoch(args, rank, epoch, model, train_dl, criterion, optimizer, printer=print):
    epoch_records = {"loss": []}
    model.train()
    with tqdm(enumerate(train_dl), total=len(train_dl)) as t:
        for bi, batch in t:
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = batch[key].cuda(rank)
            pred = model(batch)
            loss_dict = criterion(pred, batch)
            loss = loss_dict["loss"]
            (loss / args.cumulative_grad_batchs).backward()
            for key in loss_dict:
                if key not in epoch_records: epoch_records[key] = []
                epoch_records[key].append(loss_dict[key].item())
            
            if ((bi+1) % args.cumulative_grad_batchs == 0) or ((bi+1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if ((bi+1) % args.display == 0) or ((bi+1) == len(train_dl)):
                t.set_description(f"{rank} [{epoch:03d}] [{bi+1:03d}/{len(train_dl)}]")
                t.set_postfix({key: round(np.mean(value), 4) for key, value in epoch_records.items()})
             
    printer(f"*** {rank} [{epoch:03d}] {' '.join([key+'='+str(round(np.mean(value), 4)) for key, value in epoch_records.items()])}")
    return epoch_records


def run(rank, args, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    logger = build_logger(os.path.join(args.save_dir, f"log_{rank}.log"))
    printer = logger.info
    log_parser_args(args, printer)
    
    model = eval(args.model)(args).cuda(rank)
    printer(f"Total trainable parameters: {count_parameters(model):.2f}M")
    if os.path.isfile(args.checkpoint):
        model = load_checkpoint(model, args.checkpoint, args.checkpoint_prefix, printer=printer)
    model = DDP(model, device_ids=[rank])
    
    train_ds = eval(args.dataset)(args, args.train_file, split="train", printer=printer)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size//world_size, num_workers=4, pin_memory=True, sampler=train_sampler) # 32//world_size

    criterion = eval(args.criterion)(args).to(args.device)
    optimizer = build_optimizer(model, args)
    lr_scheduler = build_lr_scheduler(optimizer, args)

    epoch_lst = []
    train_epoch_records, val_epoch_records, test_epoch_records = {}, {}, {}
    # best_focus_metric = 0.0
    for epoch in range(args.epochs):
        epoch_lst.append(epoch)

        train_records = train_one_epoch(args, rank, epoch, model, train_dl, criterion, optimizer, printer=printer)
        train_records = {f"{key}_{rank}": value for key, value in train_records.items()}
        for key, value in train_records.items():
            if key not in train_epoch_records: train_epoch_records[key] = []
            train_epoch_records[key].append(np.mean(value))
        
        if rank == 0:
            plot_training_results(args.focus_metric, epoch_lst, train_epoch_records, os.path.join(args.save_dir, "plot.jpg"))
            if (epoch and (epoch % args.save_model_per_epochs == 0)) or (epoch == args.epochs - 1):
                save_checkpoint(epoch, model, optimizer, lr_scheduler, os.path.join(args.save_dir, "checkpoints", f"model_{epoch:04d}.pth"))
            save_checkpoint(epoch, model, optimizer, lr_scheduler, os.path.join(args.save_dir, "checkpoints", "model_last.pth"))
            printer("="*100)
            
        lr_scheduler.step()
        
    # cleanup()
        
def main():
    args = build_args()
    set_seed(args.seed)
    exp_name = f"{args.model}"
    args.save_dir = os.path.join(args.save_dir, exp_name, str(datetime.now().strftime("%Y%m%d%H%M")))
    os.makedirs(args.save_dir, exist_ok=True)
    world_size = torch.cuda.device_count()
    spawn(run, args=(args, world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()