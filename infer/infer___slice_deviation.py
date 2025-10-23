from infer import * 
from math import floor

def _convert_support_mask(mask_array, **kwargs):
    slice_deviation_ratio = kwargs["slice_deviation_ratio"]
    slice_deviation_direction = kwargs["slice_deviation_direction"]
    z_indexes = []
    max_area = 0
    max_index = -1
    for index, slice_ in enumerate(mask_array):
        area = np.sum(slice_ > 0)
        if area:
            z_indexes.append(index)
        if area >= max_area:
            max_area = area
            max_index = index
    z_indexes = sorted(z_indexes)
    slice_deviation_amount = floor(slice_deviation_ratio * len(z_indexes))
    center_index = z_indexes.index(max_index)
    support_index = z_indexes[min(len(z_indexes)-1, max(0, center_index + slice_deviation_direction * slice_deviation_amount))]
    support_mask_array = np.zeros_like(mask_array).astype(mask_array.dtype)
    support_mask_array[support_index] = mask_array[support_index]
    assert np.sum(support_mask_array[support_index]), f"{np.sum(support_mask_array[support_index])}"
    return support_mask_array, support_index

class SD_AbdomenCT1K(AbdomenCT1k):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_AdrenalACCKi67Seg(AdrenalACCKi67Seg):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_AutoPETCT(AutoPETCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_AutoPETPETCT(AutoPETPETCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_AMOSCT(AMOSCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_CHAOSCT(CHAOSCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_COVID19SegChallenge(COVID19SegChallenge):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_COVID19CTSeg(COVID19CTSeg):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_HCCTACESeg(HCCTACESeg):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_HECKTOR(HECKTOR):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_LNQ2023(LNQ2023):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_INSTANCE(INSTANCE):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_KiTS(KiTS):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_KiPA(KiPA):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_NSCLCPleuralEffucion(NSCLCPleuralEffusion):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_QUBIQCT(QUBIQCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task06Lung(Task06Lung):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task03Liver(Task03Liver):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task07Pancreas(Task07Pancreas):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task08HepaticVessel(Task08HepaticVessel):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task09Spleen(Task09Spleen):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task10Colon(Task10Colon):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_WORD(WORD):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_HaNSegCT(HaNSegCT):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)
        
class SD_ACDC(ACDC):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_AMOSMR(AMOSMR):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_ATLASR20(ATLASR20):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_CHAOSMR(CHAOSMR):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_ISLES(ISLES):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_MnM2(MnM2):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_NCIISBI(NCIISBI):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_PROMISE(PROMISE):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_QinProstateRepeatability(QinProstateRepeatability):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Spine(Spine):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task02Heart(Task02Heart):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task04Hippocampus(Task04Hippocampus):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_Task05Prostate(Task05Prostate):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

class SD_WMH(WMH):
    def _convert_support_mask(self, mask_array, **kwargs):
        return _convert_support_mask(mask_array, **kwargs)

def build_dataset(args, printer=print):
    try:
        ds = eval("SD_"+args.dataset.replace("_", "").replace("-", "").replace(".", "").replace(" ", ""))(instance_threshold=args.instance_threshold, printer=printer)
    except Exception as e:
        printer(f"Error: {e}")
        raise NotImplementedError(args.dataset)
    return ds

if __name__ == "__main__":
    args = build_args()
    for slice_deviation_direction in [-1, 1]:
        for slice_deviation_ratio in [0.2, 0.15, 0.1, 0.05]:
            set_seed(args.seed)

            dataset_infer_dir = os.path.join(args.infer_dir, args.dataset, f"slice_deviation_ratio_{slice_deviation_ratio}_direction_{slice_deviation_direction}")
            os.makedirs(dataset_infer_dir, exist_ok=True)
            tmp_dataset_infer_dir = os.path.join(dataset_infer_dir, "data")
            os.makedirs(tmp_dataset_infer_dir, exist_ok=True)
            logger = build_logger(os.path.join(dataset_infer_dir, "log.log"))
            printer = logger.info
            printer(f"slice_deviation_ratio={slice_deviation_ratio}, slice_deviation_direction={slice_deviation_direction}")
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
                    support_categories, support_mask_arrays, support_zs = ds.get_image_mask(
                        i, slice_deviation_ratio=slice_deviation_ratio, slice_deviation_direction=slice_deviation_direction)
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


    


        