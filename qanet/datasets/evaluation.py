import json
import os
import pickle
import shutil
import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.sparse import csr_matrix

from lib.data.evaluation.densepose_eval import DensePoseEvaluator
from lib.data.evaluation.parsing_eval import ParsingEvaluator, generate_parsing_result
from lib.utils.visualizer import Visualizer

from qanet.datasets import dataset_catalog
from PIL import Image
import glob
import time


class Evaluation(object):
    def __init__(self, cfg, training=False):
        """
        Evaluation
        :param cfg: config
        """
        self.cfg = cfg
        self.training = training
        self.iou_types = ()
        self.pet_results = {}
        self.all_iou_types = ("segm", "parsing", "keypoints", "uv")

    def parsing_eval(self, iou_type, dataset, output_folder):
        """Interface of Parsing evaluation
        """
        gt_im_dir = dataset_catalog.get_im_dir(self.cfg.TEST.DATASETS[0])
        metrics = self.cfg.PARSING.METRICS if not self.training else ['mIoU', ]
        pet_eval = ParsingEvaluator(
            dataset, self.pet_results[iou_type], gt_im_dir, output_folder, self.cfg.PARSING.SCORE_THRESH,
            self.cfg.PARSING.NUM_PARSING, metrics=metrics
        )
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mIoU = pet_eval.stats['mIoU']
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mIoU

    def coco_eval(self, iou_type, dataset, output_folder):
        """Interface of COCO evaluation
        """
        file_path = os.path.join(output_folder, iou_type + ".json")
        pet_eval = evaluate_on_coco(self.cfg, dataset.coco, self.pet_results[iou_type], file_path, iou_type)
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mAP = 0.0 if 'lvis' in self.cfg.TEST.DATASETS[0] else pet_eval.stats[0]
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mAP

    def post_processing(self, results, targets, image_ids, dataset):
        """Prepare results by preparing function of each task
        """
        num_im = len(image_ids)
        eval_results = []
        ims_results = []
        bboxes_results=[]
        prepare_funcs = []
        prepare_funcs = self.get_prepare_func(prepare_funcs)
        for prepare_func in prepare_funcs:
            prepared_result = self.prepare_results(results, targets, image_ids, dataset, prepare_func)
            if prepared_result is not None:
                assert len(prepared_result) >= 2
                eval_results.append(prepared_result[0])
                # box results include box and label
                ims_results.extend(prepared_result[1:])
                #print("eval",eval_results)
                #bboxes_results.extend(prepared_result[2])
                #print("ims_results",ims_results)
                
            else:
                eval_results.append([])
                ims_results.append([None for _ in range(num_im)])
            #print("bboxes",bboxes_results)
            
        if self.cfg.VIS.ENABLED:
            self.vis_processing(ims_results, targets, image_ids, dataset, eval_results[2:-1])
        return eval_results

    def vis_processing(self, ims_results, targets, image_ids, dataset, eval_results):
        #print("ims_results",ims_results)
        ims_masks, ims_kpts, ims_parss, det_bboxes, ims_uvs = ims_results
        
              
# =============================================================================
#         for k, target in enumerate(targets):
#             if det_bboxes[k][0] !=0 or det_bboxes[k][1:] !=0:
#                target.new_scores_img_bbox(det_bboxes[k][0],det_bboxes[k][1:]) 
# =============================================================================
           
        ims_dets_gnd = [
            np.hstack((target.im_bbox.numpy(), target.scores.numpy()[:, np.newaxis])).astype(np.float32, copy=False)
            for target in targets
        ]
        
        ims_det =[np.stack([
            np.hstack((det_bboxes[k][1:], det_bboxes[k][0]))
            for k in range (len(det_bboxes))
            ]).astype(np.float32,copy=False)]
        ims_labels = [target.labels.tolist() for target in targets]
        
        #print("det_bboxes",ims_det)
        #print("ims_det",ims_dets_gnd)
        
        for k, idx in enumerate(image_ids):
            if len(ims_det[k]) == 0:
                continue
            
            part_scores=[]
            for j in eval_results[k]:
                part_scores.append(j['part_scores'])
            
            im = dataset.pull_image(idx)
            visualizer = Visualizer(self.cfg.VIS, im, dataset=dataset)
            im_name = dataset.get_img_info(image_ids[k])['file_name']
            print("image_name",im_name)
            vis_im, vis_instances = visualizer.vis_preds(
                boxes=ims_det[k],
                classes=ims_labels[k],
                masks=ims_masks[k],
                keypoints=ims_kpts[k],
                parsings=ims_parss[k],
                uvs=ims_uvs[k],
                gnd_bboxes = ims_dets_gnd[k],
                part_scores = part_scores
            )
            
            cv2.imwrite(os.path.join(self.cfg.CKPT, 'vis_before', '{}'.format(im_name)), vis_im)
            
            # Post processing techniques    
            image_path = os.path.join(self.cfg.CKPT, 'test', 'Images','{}'.format(im_name))
            result, time1 = post_process(image_path)
            result = Image.fromarray(result, 'RGB')
            result.save(os.path.join(self.cfg.CKPT, 'vis','{}'.format(im_name)))
       
    def evaluation(self, dataset, all_results):
        """Eval results by iou types
        """
        output_folder = os.path.join(self.cfg.CKPT, 'test')
        self.get_pet_results(all_results)

        for iou_type in self.iou_types:
            if iou_type == "parsing":
                eval_result = self.parsing_eval(iou_type, dataset, output_folder)
            elif iou_type in self.all_iou_types:
                eval_result = self.coco_eval(iou_type, dataset, output_folder)
            else:
                raise KeyError("{} is not supported!".format(iou_type))
        if self.cfg.CLEAN_UP:  # clean up all the test files
            shutil.rmtree(output_folder)
        return eval_result

    def prepare_results(self, results, targets, image_ids, dataset, prepare_func=None):
        """Prepare result of each task for evaluation
        """
        if prepare_func is not None:
            return prepare_func(self.cfg, results, targets, image_ids, dataset)
        else:
            return None

    def get_pet_results(self, all_results):
        """Get preparing function of each task
        """
        all_masks, all_keyps, all_parss, all_uvs = all_results
        if self.cfg.MODEL.MASK_ON:
            self.iou_types = self.iou_types + ("segm",)
            self.pet_results["segm"] = all_masks
        if self.cfg.MODEL.KEYPOINT_ON:
            self.iou_types = self.iou_types + ("keypoints",)
            self.pet_results['keypoints'] = all_keyps
        if self.cfg.MODEL.PARSING_ON:
            self.iou_types = self.iou_types + ("parsing",)
            self.pet_results['parsing'] = all_parss
        if self.cfg.MODEL.UV_ON:
            self.iou_types = self.iou_types + ("uv",)
            self.pet_results['uv'] = all_uvs

    def get_prepare_func(self, prepare_funcs):
        """Get preparing function of each task
        """
        if self.cfg.MODEL.MASK_ON:
            prepare_funcs.append(prepare_mask_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.KEYPOINT_ON:
            prepare_funcs.append(prepare_keypoint_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.PARSING_ON:
            prepare_funcs.append(prepare_parsing_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.UV_ON:
            prepare_funcs.append(prepare_uv_results)
        else:
            prepare_funcs.append(None)

        return prepare_funcs


def prepare_mask_results(cfg, results, targets, image_ids, dataset):
    mask_results = []
    ims_masks = []

    if 'mask' not in results.keys():
        return mask_results, ims_masks

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_masks.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        bitmasks = results['mask']['ims_bitmasks'][i][:, :image_height, :image_width]
        rles = []
        for j in range(len(bitmasks)):
            # Too slow.
            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(bitmasks[j][:, :, np.newaxis], dtype=np.uint8, order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            rles.append(rle)
            
        # calculating quality scores
        mask_bbox_scores = target.scores
        mask_iou_scores = results['mask']['mask_iou_scores'][i]
        mask_pixel_scores = results['mask']['mask_pixle_scores'][i]
        alpha, beta, gamma = cfg.MASK.QUALITY_WEIGHTS
        _dot = torch.pow(mask_bbox_scores, alpha) * torch.pow(mask_iou_scores, beta) * \
               torch.pow(mask_pixel_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_masks.append(rles)
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        mask_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return mask_results, ims_masks


def prepare_keypoint_results(cfg, results, targets, image_ids, dataset):
    kpt_results = []
    ims_kpts = []

    if 'keypoints' not in results.keys():
        return kpt_results, ims_kpts

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_kpts.append(None)
            continue

        keypoints = results['keypoints']['ims_kpts'][i].numpy()
        # calculating quality scores
        kpt_bbox_scores = target.scores
        kpt_iou_scores = results['keypoints']['kpt_iou_scores'][i]
        kpt_pixle_scores = results['keypoints']['kpt_pixle_scores'][i]
        alpha, beta, gamma = cfg.KEYPOINT.QUALITY_WEIGHTS
        _dot = torch.pow(kpt_bbox_scores, alpha) * torch.pow(kpt_iou_scores, beta) * \
               torch.pow(kpt_pixle_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_kpts.append(keypoints.transpose((0, 2, 1)))
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        kpt_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "keypoints": keypoint.flatten().tolist(),
                    "score": scores[k]
                }
                for k, keypoint in enumerate(keypoints)
            ]
        )
    return kpt_results, ims_kpts


def prepare_parsing_results(cfg, results, targets, image_ids, dataset):
    pars_results = []
    ims_parss = []
    output_folder = os.path.join(cfg.CKPT, 'test')

    if 'parsing' not in results.keys():
        return pars_results, ims_parss
    #print("image_id",image_ids)
    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_parss.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        parsings = results['parsing']['ims_parsings'][i][:, :image_height, :image_width]
        # calculating quality scores
        parsing_bbox_scores = target.scores
        parsing_iou_scores = results['parsing']['parsing_iou_scores'][i]
        parsing_instance_pixel_scores = results['parsing']['parsing_instance_pixel_scores'][i]
        
        parsing_part_pixel_scores = results['parsing']['parsing_part_pixel_scores'][i]
        #print("scores part",parsing_part_pixel_scores )
        alpha, beta, gamma = cfg.PARSING.QUALITY_WEIGHTS
        instance_dot = torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta) * \
                       torch.pow(parsing_instance_pixel_scores, gamma)
        instance_scores = torch.pow(instance_dot, 1. / sum((alpha, beta, gamma))).tolist()
        part_dot = torch.stack([torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta)] *
                               (cfg.PARSING.NUM_PARSING - 1), dim=1) * torch.pow(parsing_part_pixel_scores, gamma)
        part_scores = torch.pow(part_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_parss.append(parsings.numpy())
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        parsings, instance_scores, det_bboxes,part_scores = generate_parsing_result(
            parsings, instance_scores, part_scores, parsing_bbox_scores.tolist(), semseg=None, img_info=img_info,
            output_folder=output_folder, score_thresh=cfg.PARSING.SCORE_THRESH,
            semseg_thresh=cfg.PARSING.SEMSEG_SCORE_THRESH, parsing_nms_thres=cfg.PARSING.PARSING_NMS_TH,
            num_parsing=cfg.PARSING.NUM_PARSING
        )
        pars_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "parsing": csr_matrix(parsing),
                    "score": instance_scores[k],
                    "part_scores": part_scores[k]
                }
                for k, parsing in enumerate(parsings)
            ]
        )
    #print("instance_scores",instance_scores)
    #print("det_bboxes",det_bboxes)
    return pars_results, ims_parss, det_bboxes


def prepare_uv_results(cfg, results, targets, image_ids, dataset):
    uvs_results = []
    ims_uvs = []

    if 'uv' not in results.keys():
        return uvs_results, ims_uvs

    ims_Index_UV = results['uv']['ims_Index_UV']
    ims_U_uv = results['uv']['ims_U_uv']
    ims_V_uv = results['uv']['ims_V_uv']
    h, w = ims_Index_UV[0].shape[1:]

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_uvs.append(None)
            continue
        uvs = []
        Index_UV = ims_Index_UV[i].numpy()
        U_uv = ims_U_uv[i].numpy()
        V_uv = ims_V_uv[i].numpy()

        for ind, entry in enumerate(target.im_bbox.numpy()):
            x1 = int(entry[0])
            y1 = int(entry[1])
            x2 = int(entry[2])
            y2 = int(entry[3])

            output = np.zeros([3, int(y2 - y1), int(x2 - x1)], dtype=np.float32)
            output[0] = Index_UV[ind][y1:y2, x1:x2]

            outputU = np.zeros([h, w], dtype=np.float32)
            outputV = np.zeros([h, w], dtype=np.float32)

            for part_id in range(1, cfg.UV.NUM_PATCHES + 1):
                CurrentU = U_uv[ind][part_id]
                CurrentV = V_uv[ind][part_id]
                outputU[Index_UV[ind] == part_id] = CurrentU[Index_UV[ind] == part_id]
                outputV[Index_UV[ind] == part_id] = CurrentV[Index_UV[ind] == part_id]
            output[1] = outputU[y1:y2, x1:x2]
            output[2] = outputV[y1:y2, x1:x2]
            uvs.append(output.copy())

        # calculating quality scores
        uv_bbox_scores = target.scores
        uv_iou_scores = results['uv']['uv_iou_scores'][i]
        uv_pixel_scores = results['uv']['uv_pixel_scores'][i]
        alpha, beta, gamma = cfg.UV.QUALITY_WEIGHTS
        _dot = torch.pow(uv_bbox_scores, alpha) * torch.pow(uv_iou_scores, beta) * \
               torch.pow(uv_pixel_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_uvs.append(uvs)
        for uv in uvs:
            uv[1:3, :, :] = uv[1:3, :, :] * 255
        box_dets = target.im_bbox.int()
        xs = box_dets[:, 0].tolist()
        ys = box_dets[:, 1].tolist()
        ws = (box_dets[:, 2] - box_dets[:, 0]).tolist()
        hs = (box_dets[:, 3] - box_dets[:, 1]).tolist()
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        uvs_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "uv": uv.astype(np.uint8),
                    "bbox": [xs[k], ys[k], ws[k], hs[k]],
                    "score": scores[k]
                }
                for k, uv in enumerate(uvs)
            ]
        )

    return uvs_results, ims_uvs


def evaluate_on_coco(cfg, coco_gt, coco_results, json_result_file, iou_type):
    if iou_type != "uv":
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)
        if cfg.MODEL.HIER_ON and iou_type == "bbox":
            box_results = get_box_result(cfg)
            coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
            coco_gt = coco_gt.loadRes(box_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type, True)
        else:
            if 'lvis' in cfg.TEST.DATASETS[0]:
                from lvis import LVIS, LVISResults, LVISEval
                lvis_gt = LVIS(dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0]))
                lvis_results = LVISResults(lvis_gt, coco_results)
                coco_eval = LVISEval(lvis_gt, lvis_results, iou_type)
            else:
                coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
                coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    else:
        pkl_result_file = json_result_file.replace('.json', '.pkl')
        with open(pkl_result_file, 'wb') as f:
            pickle.dump(coco_results, f, 2)
        if cfg.TEST.DATASETS[0].find('test') > -1:
            return
        eval_data_dir = cfg.DATA_DIR + '/coco/annotations/DensePoseData/eval_data/'
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = DensePoseEvaluator(coco_gt, coco_dt, iou_type, eval_data_dir, calc_mode=cfg.UV.CALC_MODE)
    return coco_eval


def binary_mask_to_color_mask(binary_mask, colors):
    h, w = binary_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    binary_array = np.array(binary_mask)
    instances = np.unique(binary_array)
    instances = instances[instances != 0]

    for i, instance in enumerate(instances):
        color = colors[instance]
        color_mask[binary_array == instance] = color

    return color_mask


def dict2array(colordict):
    keys = colordict.keys()
    colorarray = np.zeros((len(keys), 3))
    for ids, k in enumerate(keys):
        colorarray[ids] = np.asarray(colordict[k])
    colorarray = np.asarray(colorarray, dtype=np.int)

    return colorarray


def post_process(img_path):
    name = img_path[-26:-4]
    #print("image",name)
    img_path1 = (os.path.normpath(img_path + os.sep + os.pardir))

    inst_path = os.path.join(os.path.normpath(img_path1 + os.sep + os.pardir), 'instance_parsing/', name)
    seg_path = os.path.join(os.path.normpath(img_path1 + os.sep + os.pardir), 'instance_segmentation/', name)
    glob_path = os.path.join(os.path.normpath(img_path1 + os.sep + os.pardir), 'global_parsing/', name)
    

    par_score = np.loadtxt(inst_path + '.txt', usecols=(1))
    par_class = np.loadtxt(inst_path + '.txt', dtype='int', usecols=(0))
    inst_bbox = np.loadtxt(seg_path + '.txt', dtype='int', usecols=(1, 2, 3, 4))
    inst_score = np.loadtxt(seg_path + '.txt', usecols=(0))
    
    #print("scores", par_score)
    #print("scores", inst_score)

    inst_id = Image.open(inst_path + '.png')
    inst_id = np.array(inst_id)
    seg_id = Image.open(seg_path + '.png')
    seg_id = np.array(seg_id)
    glob_id = Image.open(glob_path + '.png')
    glob_id = np.array(glob_id)
    
    start = time.time() 
    
    uniques = np.unique(inst_id)
    uniques = uniques[uniques != 0]
    
    if inst_bbox.ndim == 1:
        inst_bbox = np.expand_dims(np.array(inst_bbox), axis=0)
    if np.count_nonzero(par_score) == 1:
        par_score = np.expand_dims(np.array(par_score), axis=0)
    if np.count_nonzero(inst_score) == 1:
        inst_score = np.expand_dims(np.array(inst_score), axis=0)
    if np.count_nonzero(par_class) == 1:
        par_class = np.expand_dims(np.array(par_class), axis=0)

    # Getting instances
    instances_with_classes = ((seg_id - 1) * 20) + glob_id
    instances_with_classes = np.where(instances_with_classes > 225, 0, instances_with_classes)
    instances_with_classes = np.where(instances_with_classes%20 == 0, 0, instances_with_classes)

     # Handling all instances
    instances_with_classes_unique = np.unique(instances_with_classes)[1:]
    # print("uniques", instances_with_classes_unique)
    h, w = inst_id.shape
    img = np.zeros((h, w), dtype=int)

    for instance in np.unique(seg_id)[1:]:  # excluding zeros
        instance_score = inst_score[instance - 1]

        classes_present = []
        class_scores = []
        parsing_val =[]
        centers_instances = []
        classes_instances = []
        
        for ix, val in enumerate(instances_with_classes_unique):
            if int(val / 20) == (instance - 1 ) and ix < len(par_score):
                classes_present.append(val % 20)
                class_scores.append(par_score[ix])
                parsing_val.append(val)
        
        
        img_instance = np.where(seg_id == instance, instance, 0)
        img1 = np.zeros((h, w), dtype=int)
        image_dummy2 = np.zeros((h, w), dtype=int)

        for id, (k, class_k) in enumerate(zip(parsing_val, classes_present)):
            
           
            image_dummy = np.where(instances_with_classes == k, k, 0)
            if class_k == 1:
                if np.count_nonzero(image_dummy) > (np.count_nonzero(img_instance)/2):
                    image_dummy = np.where(image_dummy > 0, k, 0)                  
                else:
                    image_dummy = np.where(image_dummy > 0, 0, 0)

            # Remove low scores of classes
            if (class_scores[id] < 0.65 and class_k != 7)  :
                image_dummy = np.where(image_dummy == k, 0, image_dummy) 
                
                    
            if instance_score <= 0.96:
                contours_large = []
                image_dummy1 = np.zeros((h, w), dtype='uint8')
                contours, _ = cv2.findContours(np.array(image_dummy, dtype='uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                if class_k != 7:
                    contours_large = []
                    for contour in contours:
                        if cv2.contourArea(contour) >= 1000:
                            contours_large.append(contour)
                else:
                    for contour in contours:
                        contours_large.append(contour)

                if contours_large != []:
                    sorted_contours = sorted(contours_large, key=cv2.contourArea, reverse=True)
                    largest_contor = [sorted_contours[0]]
                else:
                    largest_contor = contours_large

                image_dummy = cv2.drawContours(image_dummy1, largest_contor, -1,
                                                          (int(k)), cv2.FILLED)
                image_dummy = cv2.bitwise_and(np.array(image_dummy,dtype='uint8'),
                                              np.array(image_dummy, dtype='uint8'), mask= np.array(img_instance,dtype='uint8'))

            img1 = img1 + image_dummy
            
            if instance_score <= 0.96:
                for i in (np.unique(image_dummy)):
                    if i > 0:
                       centers_instances.append([largest_contor])
                       classes_instances.append(i)

        img = img + img1
        
        if instance_score <= 0.96:
            neg_img_instance = np.where(img_instance > 0, 1, 0)
            neg_img = np.where(img > 0, 1, 0)
            neg_contours = cv2.bitwise_xor(np.array(neg_img_instance, dtype='uint8'), np.array(neg_img, dtype='uint8'), mask=np.array(neg_img_instance, dtype='uint8'))

            contours, _ = cv2.findContours(np.array(neg_contours, dtype='uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                point = []
                if M['m00'] != 0:
                   cx = int(M['m10'] / M['m00'])
                   cy = int(M['m01'] / M['m00'])
                   point = [cx, cy]

                if len(centers_instances) > 0 and point != [] :
                   min_distances=[]
                   
                   for vals in centers_instances:
                       
                       vals = np.squeeze(vals)  
                                                           
                       if vals.ndim == 1:
                          vals = np.expand_dims(np.array(vals), axis=0)
                       
                       distances = np.linalg.norm(vals - np.array(point), axis=1)
                       min_distances.append(min(distances))
                   min_distances1 = np.argmin(min_distances)
                   contour_color = classes_instances[min_distances1]
                else:
                   contour_color = 0
  
                if np.count_nonzero(contour) > 800:
                    
                    if contour_color > 10:
                        in_id = int(contour_color/ 20 ) - 1
                    else:
                        in_id = 0
                    
                    if len(centers_instances) == 1 and contour_color % 20 == 2:
                        contour_color = int(in_id * 20) + 7
                    elif len(centers_instances) == 1 and contour_color % 20 == 7:
                        contour_color = int(in_id * 20) + 3
                    elif len(centers_instances) == 2 and (contour_color % 20 == 5 or contour_color % 20 == 6) :
                        contour_color = int(in_id * 20) + 4
                    elif len(centers_instances) == 3 and contour_color % 20 == 6:
                        contour_color = int(in_id * 20) + 5
                    else:
                        contour_color = contour_color
                        
                # Redundant values
                if contour_color % 20 > 7:
                    contour_color = (contour_color / 20) + 7
                
                image_dummy2 = cv2.drawContours(np.array(image_dummy2, dtype='uint8'), [contour], -1, (int(contour_color)), cv2.FILLED)
                image_dummy2 = cv2.bitwise_and(np.array(image_dummy2, dtype='uint8'),np.array(image_dummy2, dtype='uint8'), mask=np.array(neg_img_instance, dtype='uint8')) 
            
                   
        img = img + image_dummy2
        
        
    instances_with_classes = img
    inst_masks = instances_with_classes % 20
    inst_masks = np.where(inst_masks % 20 > 7, 0,inst_masks) 

    colormap = {0: [0, 0, 0], 1: [0, 255, 255], 2: [0, 0, 255], 3: [0, 85, 0], 4: [255, 255, 0], 5: [0, 85, 255],
                6: [85, 0, 0], 7: [255, 0, 255], 8: [221, 0, 0], 9: [85, 85, 0], 10: [0, 51, 85],
                11: [128, 86, 52], 12: [0, 128, 0], 13: [255, 0, 0], 14: [0, 85, 85], 15: [51, 0, 170],
                16: [221, 119, 0], 17: [85, 255, 170], 18: [0, 0, 128], 19: [0, 170, 255]}

    colormap = dict2array(colormap)
    colormap[:, [2, 0]] = colormap[:, [0, 2]]

    color_mask = binary_mask_to_color_mask(inst_masks, colormap)
    
    image = cv2.imread(img_path)
    # print("path", img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = cv2.addWeighted(image, 1.0, color_mask.astype('uint8'), 0.5, 0, dtype=cv2.CV_8UC1)
    
    for val, i in enumerate(inst_bbox):
        x1,y1,x2,y2 = i
        cv2.rectangle(result, (int(y1),int(x1)), (int(y2),int(x2)), (0,255,0), 2)
        
        txt = "Raw Cutting: " 
        back_tl = int(y1), int((x1) - 25)
        back_br = int(y1 )+ int(len(txt)*10), int(x1)
        cv2.rectangle(result, back_tl, back_br, (170,170,170), -1)
        cv2.putText(result, txt + str(inst_score[val])[:4], (y1, x1 - 12), 0, 0.45, (0,0,0), 1)
    
    end = time.time() 
    total_time = end - start
    
    inst_path1 = os.path.join(os.path.normpath(img_path1 + os.sep + os.pardir), 'instance_parsing/', name)
    glob_path1 = os.path.join(os.path.normpath(img_path1 + os.sep + os.pardir), 'global_parsing/', name)

    values = np.unique(instances_with_classes)[1:]
    par_val_final =[]
    
    dict1={}
    for dic_key, dic_val in enumerate(instances_with_classes_unique):
        dict1[dic_val]=dic_key
    
    for f in values:
        if f in dict1.keys():
           par_val_final.append((f % 20, par_score[dict1[f]]))
            
    instances_with_classes1 = instances_with_classes
    
    for num1, val1 in enumerate(values):
        instances_with_classes1 = np.where(instances_with_classes1==val1, num1+1, instances_with_classes1)
    
    # print("size", instances_with_classes1)
    inst_par_img = Image.fromarray(instances_with_classes1.astype(np.uint8))
    inst_par_img.save(inst_path1+'.png')
    np.savetxt(inst_path1+'.txt', par_val_final, fmt='%i %1.16f')
    
    inst_par_img11 = instances_with_classes % 20
    inst_par_img11 = np.where(inst_par_img11 > 7, 0, inst_par_img11)
    inst_par_img1 = Image.fromarray((inst_par_img11).astype(np.uint8))
    inst_par_img1.save(glob_path1+'.png')
    
    return result, total_time
