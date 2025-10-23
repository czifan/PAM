from imports import *

def train_one_epoch(args, epoch, model, train_dl, criterion, optimizer, printer=print):
    epoch_records = {"loss": []}
    model.train()
    with tqdm(enumerate(train_dl), total=len(train_dl)) as t:
        for bi, batch in t:
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = batch[key].to(args.device)

            pred = model(batch)
            loss_dict = criterion(pred, batch)
            loss = loss_dict["loss"]
            (loss / args.cumulative_grad_batchs).backward()
            
            for key in loss_dict:
                if key not in epoch_records: epoch_records[key] = []
                epoch_records[key].append(loss_dict[key].item())

            # metric_dict = compute_metric(pred, batch)
            # for key in metric_dict:
            #     if key not in epoch_records: epoch_records[key] = []
            #     epoch_records[key].append(metric_dict[key])

            if ((bi+1) % args.cumulative_grad_batchs == 0) or ((bi+1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if ((bi+1) % args.display == 0) or ((bi+1) == len(train_dl)):
                t.set_description(f"[{epoch:03d}] [{bi+1:03d}/{len(train_dl)}]")
                t.set_postfix({key: round(np.mean(value), 4) for key, value in epoch_records.items()})
             
    printer(f"*** [{epoch:03d}] {' '.join([key+'='+str(round(np.mean(value), 4)) for key, value in epoch_records.items()])}")
    return epoch_records
            

def main():
    args = build_args()
    set_seed(args.seed)
    
    exp_name = f"{args.model}"
    args.save_dir = os.path.join(args.save_dir, exp_name, str(datetime.now().strftime("%Y%m%d%H%M")))
    os.makedirs(args.save_dir, exist_ok=True)

    logger = build_logger(os.path.join(args.save_dir, "log.log"))
    printer = logger.info
    log_parser_args(args, printer)

    model = eval(args.model)(args) 
    printer(f"Total trainable parameters: {count_parameters(model):.2f}M")
    if os.path.isfile(args.checkpoint):
        model = load_checkpoint(model, args.checkpoint, args.checkpoint_prefix, printer=printer)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        printer(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(args.device)
    
    train_ds = eval(args.dataset)(args, args.train_file, split="train", printer=printer)
#     for i, batch in enumerate(train_ds):
#         if i > 100: break
#         show_task(batch["support_x"], batch["support_y"], batch["query_xs"], batch["query_ys"], f"debug/{batch['modality']}_{batch['sample_id'].replace('/', '+')}.jpg")
#     return
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=16, pin_memory=True)        

    criterion = eval(args.criterion)(args).to(args.device)
    optimizer = build_optimizer(model, args)
    lr_scheduler = build_lr_scheduler(optimizer, args)

    epoch_lst = []
    train_epoch_records = {}
    best_focus_metric = 0.0
    for epoch in range(args.epochs):
        epoch_lst.append(epoch)
        train_records = train_one_epoch(args, epoch, model, train_dl, criterion, optimizer, printer=printer)
        for key, value in train_records.items():
            if key not in train_epoch_records: train_epoch_records[key] = []
            train_epoch_records[key].append(np.mean(value))
        lr_scheduler.step()
        plot_training_results(args.focus_metric, epoch_lst, train_epoch_records, os.path.join(args.save_dir, "plot.jpg"))
        if (epoch and (epoch % args.save_model_per_epochs == 0)) or (epoch == args.epochs - 1):
            save_checkpoint(epoch, model, optimizer, lr_scheduler, os.path.join(args.save_dir, "checkpoints", f"model_{epoch:04d}.pth"))
        save_checkpoint(epoch, model, optimizer, lr_scheduler, os.path.join(args.save_dir, "checkpoints", "model_last.pth"))
        #     torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{epoch:04d}.pth"))
        # torch.save(model.state_dict(), os.path.join(args.save_dir, "model_last.pth"))
        printer("="*100)

if __name__ == "__main__":
    main()