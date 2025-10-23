from infer import * 
from math import floor

if __name__ == "__main__":
    args = build_args()

    for slice_tickness in [10, 30, 40]:
        args.neighbor_spacing_z = slice_tickness
        set_seed(args.seed)

        dataset_infer_dir = os.path.join(args.infer_dir, args.dataset, f"slice_tickness_{slice_tickness}")
        os.makedirs(dataset_infer_dir, exist_ok=True)
        tmp_dataset_infer_dir = os.path.join(dataset_infer_dir, "data")
        os.makedirs(tmp_dataset_infer_dir, exist_ok=True)
        logger = build_logger(os.path.join(dataset_infer_dir, "log.log"))
        printer = logger.info
        printer(f"slice_tickness={slice_tickness}")
        log_parser_args(args, printer)

        propnet = eval(args.model)(args) 
        printer(f"Total trainable parameters: {1.0*count_parameters(propnet)/1e6:.2f}M")
        propnet = load_checkpoint(propnet, args.checkpoint, printer=print)
        propnet = propnet.to(args.device)

        if args.box2seg_model in ["Box2SegNet",]:
            box2seg = eval(args.box2seg_model)(args)
            printer(f"[Seg2box] Total trainable parameters: {1.0*count_parameters(box2seg)/1e6:.2f}M")
            box2seg = load_checkpoint(box2seg, args.box2seg_checkpoint, printer=print)
            box2seg = box2seg.to(args.device)
        else:
            box2seg = None

        num_ignored = 0
        records = {"dice": []}
        ds = build_dataset(args, printer=printer)
        for i in range(len(ds)):
            patient_id, img_file, img, img_array, spacing, mask, mask_array, support_category_ids, \
                support_categories, support_mask_arrays, support_zs = ds.get_image_mask(i)
            if img is None:
                printer(f"img is None")
                printer("=" * 100)
                num_ignored += 1
                continue
            if len(support_mask_arrays) == 0: 
                printer(f"!!! Ignored, there are not targets ({len(support_mask_arrays)}), {img_file}")
                printer("=" * 100)
                num_ignored += 1
                continue
            used_time, pred_array = infer(args, ds.get_modality(), box2seg, propnet, img, img_array, spacing, 
                    mask, support_category_ids, support_categories, support_mask_arrays, support_zs, 
                    None, args.target_size, args.dynamic_crop_ratio, args.neighbor_spacing_z, args.max_Z, args.device, printer)
            sample_metric_dict = calculate_metrics(pred_array, mask_array, ds.labels_dict)
            sample_metric_dict["used_time"] = used_time
            for key in sample_metric_dict:
                if key not in records:
                    records[key] = []
                records[key].append(sample_metric_dict[key])
                if "dice" in key:
                    records["dice"].append(sample_metric_dict[key])
            printer(f"\tsample={sample_metric_dict}")
            print_context = "\t".join(f"{key}={np.mean(value):.4f}" for key, value in records.items())
            printer(f"{i}\t{print_context}")

            save_name = img_file.replace("/", "___").split(".")[0] + "___" + patient_id + "___" + str(int(i))
            printer(f"Save to {save_name}")
            img_file = ds._read_image_mask(i)[0]
            img_nii = sitk.GetImageFromArray(img_array)
            img_nii.SetSpacing(spacing)
            sitk.WriteImage(img_nii, os.path.join(tmp_dataset_infer_dir, save_name+".img.nii.gz"))
            pred_nii = sitk.GetImageFromArray(pred_array)
            pred_nii.SetSpacing(spacing)
            sitk.WriteImage(pred_nii, os.path.join(tmp_dataset_infer_dir, save_name+".pred.nii.gz"))
            mask_nii = sitk.GetImageFromArray(mask_array)
            mask_nii.SetSpacing(spacing)
            sitk.WriteImage(mask_nii, os.path.join(tmp_dataset_infer_dir, save_name+".mask.nii.gz"))
            printer("=" * 100)

        printer(f"A total of {ds.__len__()} samples !!! Ignored {num_ignored} samples !!!")
        release_logger(logger)


    


        