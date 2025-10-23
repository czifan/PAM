from train import *

def evaluate_one_epoch(args, epoch, model, eval_dl, criterion, printer=print):
    epoch_records = {"loss": []}
    model.eval()
    with tqdm(enumerate(eval_dl), total=len(eval_dl)) as t:
        for bi, batch in t:
            with torch.no_grad():
                for key in batch:
                    if not isinstance(batch[key], list):
                        batch[key] = batch[key].to(args.device)
                pred = model(batch)
                loss_dict = criterion(pred, batch)

            for key in loss_dict:
                if key not in epoch_records: epoch_records[key] = []
                epoch_records[key].append(loss_dict[key].item())

            metric_dict = compute_metric(pred, batch)
            for key in metric_dict:
                if key not in epoch_records: epoch_records[key] = []
                epoch_records[key].append(metric_dict[key])

            if ((bi+1) % args.display == 0) or ((bi+1) == len(eval_dl)):
                t.set_description(f">[{epoch:03d}] [{bi+1:03d}/{len(eval_dl)}]")
                t.set_postfix({key: round(np.mean(value), 4) for key, value in epoch_records.items()})

    epoch_records = dict(sorted(epoch_records.items(), key=lambda x: x[0]))
    printer(f"> *** [{epoch:03d}] {' '.join([key+'='+str(round(np.mean(value), 4)) for key, value in epoch_records.items()])}")
    return epoch_records

def main():
    args = build_args()
    set_seed(args.seed)
    
    logger = build_logger(os.path.join(args.save_dir, f"{os.path.basename(args.eval_file).split('.')[0].lower()}_{os.path.basename(args.checkpoint).split('.')[0].split('_')[1]}.log"))
    printer = logger.info
    log_parser_args(args, printer)

    model = eval(args.model)(args) 
    model = load_checkpoint(model, args.checkpoint, printer=print)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        printer(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(args.device)

    criterion = eval(args.criterion)(args).to(args.device)

    eval_ds = eval(args.dataset)(args, args.eval_file, split="eval", printer=printer)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    evaluate_one_epoch(args, -1, model, eval_dl, criterion, printer=printer)
    
if __name__ == "__main__":
    main()