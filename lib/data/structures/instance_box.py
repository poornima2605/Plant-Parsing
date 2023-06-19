import torch

from lib.data.structures.instance import get_affine_transform_modified
import os
import cv2
import numpy as np
from lib.ops import roi_align_rotated
from PIL import Image

class InstanceBox(object):
    def __init__(self, bbox, labels, image_size, scores=None, ori_bbox=None, im_bbox=None, parsing=None):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        labels = torch.as_tensor(labels)
        scores = torch.as_tensor(scores) if scores is not None else torch.ones(len(bbox))
        ori_bbox = torch.as_tensor(ori_bbox) if ori_bbox is not None else bbox
        if im_bbox is not None:
            im_bbox = torch.as_tensor(im_bbox)
        else:
            xmin, ymin, w, h= bbox.split(1, dim=-1) # xywh t0 xyxy
            im_bbox = torch.cat(
                (xmin, ymin, xmin + w, ymin + h), dim=-1
            )
        if bbox.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(bbox.ndimension()))
        if bbox.size(-1) != 4 and bbox.size(-1) != 5:
            raise ValueError("last dimension of bbox should have a size of 4 or 5, got {}".format(bbox.size(-1)))

        self.bbox = bbox
        self.labels = labels
        self.scores = scores
        self.ori_bbox = ori_bbox
        self.im_bbox = im_bbox
        self.size = image_size # (w, h)
        self.extra_fields = {}
        self.trans = None
        self.parsing=[]
        
        if parsing != None and not(torch.is_tensor(parsing)) :
            for i in parsing:
                self.parsing.append(Parsing_test(i))
            self.parsing=torch.as_tensor(self.parsing)
        else:
            self.parsing = parsing
        

    def convert(self, aspect_ratio, scale_ratio):
        bbox = self.bbox
        
        # bbox[:, 2]= (bbox[:, 2] - bbox[:,0])
        # bbox[:, 3]= (bbox[:, 3] - bbox[:,1])
        bbox[:, 0] += (bbox[:, 2]) / 2.0
        bbox[:, 1] += (bbox[:, 3]) / 2.0
        xc, yc, w, h = bbox.split(1, dim=-1)
        
        h[w > aspect_ratio * h] = w[w > aspect_ratio * h] * 1.0 / aspect_ratio
        w[w < aspect_ratio * h] = h[w < aspect_ratio * h] * aspect_ratio
        w *= 1.1
        h *= 1.1
        rot = torch.zeros(xc.shape).to(dtype=xc.dtype)
        self.ori_bbox = torch.cat((xc, yc, w, h, rot), dim=-1)
        self.bbox = self.ori_bbox #* scale_ratio

        xmin, ymin, w, h, _ = self.bbox.split(1, dim=-1) # xywh t0 xyxy
        self.im_bbox = torch.cat(
                (xmin, ymin, xmin + w, ymin + h), dim=-1
            )

    def crop_and_resize(self, train_size, affine_mode='cv2'):
        parsing=[]
        if affine_mode == 'cv2':
            self.trans=[]      
            for i, box in zip(self.parsing, self.bbox):
                trans = get_affine_transform_modified(box, train_size)
                self.trans.append(trans)
                parsing.append(crop_and_resize(box, train_size, trans, i))
        else:
            for i, box in zip(self.parsing, self.bbox):
                parsing.append(crop_and_resize(box, train_size, self.trans, i))

        parsing_numpy = np.array([pars.numpy() for pars in parsing],dtype=np.int64)
        # for i in parsing_numpy:
        #     im = Image.fromarray(np.uint8(i*32))
        #     j = np.random.randint(1,100)
        #     im.save('/home/student3/anaconda3/envs/QANet/QANet/data/images_before_testing/'+str(j)+'.png')
        self.parsing=torch.as_tensor(parsing_numpy)
    
    def new_scores_img_bbox(self, scores, bboxes):
        self.scores = torch.as_tensor(scores)
        bbox= torch.as_tensor(bboxes)
        xmin, ymin, xmax, ymax = bbox.split(1, dim=-1) # xywh t0 xyxy
        self.im_bbox = torch.cat(
                (xmin, ymin, xmax, ymax), dim=-1
            )

    def __len__(self):
        return self.bbox.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
    
    def __call__(self):
        return self.bbox


def instancebox_split(instancebox, batch_size):
    bbox = instancebox.bbox.split(batch_size, dim=0)
    labels = instancebox.labels.split(batch_size, dim=0)
    scores = instancebox.scores.split(batch_size, dim=0)
    ori_bbox = instancebox.ori_bbox.split(batch_size, dim=0)
    im_bbox = instancebox.im_bbox.split(batch_size, dim=0)
    parsing = instancebox.parsing.split(batch_size, dim=0)
    image_size = instancebox.size
    results = [InstanceBox(_bbox, _labels, image_size, _scores, _ori_bbox, _im_bbox, _im_parsing)
               for _bbox, _labels, _scores, _ori_bbox, _im_bbox, _im_parsing
               in zip(bbox, labels, scores, ori_bbox, im_bbox, parsing)]
    return results


def Parsing_test(parsing_list):
    root_dir, file_name, parsing_id = parsing_list
    human_dir = root_dir.replace('Images', 'Human_ids')
    category_dir = root_dir.replace('Images', 'Category_ids')
    file_name = file_name.replace('jpg', 'png')
    human_path = os.path.join(human_dir, file_name)
    category_path = os.path.join(category_dir, file_name)
    human_mask = cv2.imread(human_path, 0)
    category_mask = cv2.imread(category_path, 0)
    parsing = category_mask * (human_mask == parsing_id)
    #im = Image.fromarray(np.uint8(parsing*255))
    #im.save('/home/student3/anaconda3/envs/QANet/QANet/data/images_before_testing/'+str(file_name[:-4])+str(parsing_id)+'.png')
    return parsing
        
        # size=400,400
        # new_image=Image.fromarray(numpy.uint8(parsing)).convert('L')        
        # new_pixels_value=Image.fromarray(numpy.array(new_image)*32)
        # #new_pixels_value.save('/home/student3/anaconda3/envs/QANet/QANet/data/results1/'+ file_name + '.png')
        # new_pixels_value.thumbnail(size, Image.ANTIALIAS)
        # new_pixels_value.show()
        # # display image for 10 seconds
        # time.sleep(10)
        # # hide image
        # for proc in psutil.process_iter():
        #     if proc.name() == "display":
        #        proc.kill() 

def crop_and_resize(bbox, train_size, trans, parsing):
    
    if trans is None:
        parsing = torch.from_numpy(np.ascontiguousarray(parsing)).to(dtype=torch.float32)
        bbox = bbox[None]
        batch_inds = torch.tensor([0.])[None]
        rois = torch.cat([batch_inds, bbox], dim=1)  # Nx5

        parsing = roi_align_rotated(
            parsing[None, None], rois, (train_size[1], train_size[0]), 1.0, 0, True).squeeze()
            
    else:
        parsing = cv2.warpAffine(
        np.float32(parsing),
        trans,
        (int(train_size[0]), int(train_size[1])),
        flags=cv2.INTER_NEAREST
        )
        parsing = torch.from_numpy(parsing)
    
    return parsing
               
    
