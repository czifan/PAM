from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
from torch.cuda.amp import GradScaler, autocast
from imports import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def eval_one_epoch(args, rank, epoch, model, eval_dl, criterion, printer=print):
    epoch_records = {"loss": []}
    model.eval()
    with tqdm(enumerate(eval_dl), total=len(eval_dl)) as t:
        for bi, batch in t:
            with torch.no_grad():
                for key in batch:
                    if not isinstance(batch[key], list):
                        batch[key] = batch[key].cuda(rank)
                pred = model(batch)
                loss_dict = criterion(pred, batch)
                loss = loss_dict["loss"]
            for key in loss_dict:
                if key not in epoch_records: epoch_records[key] = []
                epoch_records[key].append(loss_dict[key].item())

            if ((bi+1) % args.display == 0) or ((bi+1) == len(eval_dl)):
                t.set_description(f"{rank} [{epoch:03d}] [{bi+1:03d}/{len(eval_dl)}]")
                t.set_postfix({key: round(np.mean(value), 4) for key, value in epoch_records.items()})
             
    printer(f"*** {rank} [{epoch:03d}] {' '.join([key+'='+str(round(np.mean(value), 4)) for key, value in epoch_records.items()])}")
    return epoch_records

def run(rank, args, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    split = os.path.basename(args.eval_file).split(".")[0].lower()

    logger = build_logger(os.path.join(args.save_dir, f"{split}_{rank}.log"))
    printer = logger.info
    log_parser_args(args, printer)
    
    eval_ds = eval(args.dataset)(args, args.eval_file, split=split, printer=printer)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size//world_size, num_workers=2, pin_memory=True, sampler=eval_sampler) # 32//world_size

    criterion = eval(args.criterion)(args).to(args.device)

    epoch_lst = []
    eval_epoch_records = {}
    # best_focus_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs+1, args.save_model_per_epochs):
        epoch_lst.append(epoch)

        model = eval(args.model)(args).cuda(rank)
        printer(f"Total trainable parameters: {count_parameters(model):.2f}M")
        checkpoint_file = os.path.join(args.checkpoint, f"model_{epoch:04d}.pth")
        assert os.path.isfile(checkpoint_file), checkpoint_file
        model = load_checkpoint(model, checkpoint_file, args.checkpoint_prefix, printer=printer)
        model = DDP(model, device_ids=[rank])

        eval_records = eval_one_epoch(args, rank, epoch, model, eval_dl, criterion, printer=printer)
        eval_records = {f"{key}_{rank}": value for key, value in eval_records.items()}
        for key, value in eval_records.items():
            if key not in eval_epoch_records: eval_epoch_records[key] = []
            eval_epoch_records[key].append(np.mean(value))
        
        if rank == 0:
            plot_training_results(args.focus_metric, epoch_lst, eval_epoch_records, os.path.join(args.save_dir, f"{split}_plot.jpg"), split=split)
            printer("="*100)
        
def main():
    args = build_args()
    set_seed(args.seed)
    world_size = torch.cuda.device_count()
    spawn(run, args=(args, world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()