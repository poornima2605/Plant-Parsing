import json
import pycocotools.mask
import PIL
import numpy
import matplotlib
import tqdm
import datetime
import sys, getopt, os, shutil
from PIL import Image


import numpy as np
import cv2



def rle_decode(mask_rle, shape):

    starts, lengths = [numpy.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts = starts - 1
    ends = starts + lengths
    img = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def main(argv):
    in_path = ''
    out_path = ''
    superclass = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:s:", ["annotation_file=", "input_folder", "output_folder=", "superclass"])

        if len(opts)< 3:
            print('usage: python Hasty_to_COCO.py -i <input_file> -o <output_file> -s <supercategory>')
            sys.exit(2)

    except getopt.GetoptError:
        print('python Hasty_to_COCO.py -i <input_file> -o <output_file> -s <supercategory>')
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            print('python Hasty_to_COCO.py -i <input_file> -o <output_file> -s <supercategory>')
            sys.exit()

        elif opt in ("-i", "--input_file"):
            in_path = arg

        elif opt in ("-o", "--output_file"):
            out_path = arg

        elif opt in ("-s", "--supercategory"):
            superclass = arg

    if not os.path.exists(in_path):
        print('Input file could not be located.')
        sys.exit(3)

    out_file = {
        "info": {
            "description": "GreenAI Dataset",
            "url": None,
            "version": "1.0",
            "year": 2021,
            "contributor": None,
            "date_created": datetime.date.today().strftime('%y/%m/%d')
            },
        "licenses": [
            {
                "url": None,
                "id": 1,
                "name": None
                }
            ],
        "images": list(),
        "annotations": list(),
        "categories": list()
        }

    images_sect_tpl = list()
    image_subsect_tpl = {
        "license": 1,
        "id": None,
        "file_name": None,
        "width": None,
        "height": None,
        }
    annotations_sect_tpl = list()
    annotation_subsect_tpl = {
        "area": None,
        "bbox": list(),
        "segmentation": [],
        "category_id": None,
        "id": None,
        "image_id": None,
        "iscrowd": None,
        "parsing_id": None
        }
    categories_sect_tpl = list()
    category_subsect_tpl = {
        "parsing": dict(),
        "id": None,
        "name": None,
        "supercategory": superclass
        }

    images_sect = images_sect_tpl
    image_subsect = image_subsect_tpl
    annotations_sect = annotations_sect_tpl
    annotation_subsect = annotation_subsect_tpl
    categories_sect = categories_sect_tpl
    category_subsect = category_subsect_tpl

    class_id_asgmt = dict()

    print('Opening input json file: {}'.format(in_path))

    with open(in_path) as in_file_handle:

        in_file = json.load(in_file_handle)  # Reading json file

        label_classes = in_file["label_classes"]
        class_idx = 0
        for label_class  in  label_classes :

            #if label_class["class_name"] == "Raw Cutting":
            if  label_class["class_name"] == "Raw Cutting" or label_class["class_name"] == "Tip Cutting" or label_class["class_name"] == "First Section Cutting" \
                       or label_class["class_name"] == "Redundant Top End" or label_class["class_name"] == "Redundant Bottom End" \
                       or label_class["class_name"] == "Second Section Cutting" \
                        or label_class["class_name"] == "Third Section Cutting"or \
                        label_class["class_name"] == "Non-Viable Part": # or label_class["class_name"] == "Remains":

                #class_id_asgmt[label_class["class_name"]] = class_idx
                
                if label_class["class_name"]!= superclass:
                    
                    class_id_asgmt[class_idx]=label_class["class_name"]
                    
                elif label_class["class_name"]== superclass:
                    
                    class_id_asgmt[class_idx]="Background"

                #if label_class["class_name"]!= superclass:
                category_subsect["parsing"] = class_id_asgmt
                category_subsect["name"] = label_class["class_name"]
                category_subsect["id"] = class_idx + 1

                if (label_class["class_name"]== superclass):
                    
                    categories_sect.append(category_subsect.copy())

                class_idx = class_idx +1

        # Accessing the 'images' section of the JSON file
        imgs = in_file['images']

        # Keeping only the unique objects in the list of dictionaries
        uqe_imgs = {x['image_id']:x for x in imgs}.values()

        # Keeping the user up to date on found number of unique images and distinct object labels
        print('Number of unique images in file: {}'.format(len(uqe_imgs)))
        print('Number of distinct objects in file: {}'.format(sum([len(x['labels']) for x in uqe_imgs])))

        # Introducing some counters for indexing duties
        img_idx = 1
        obj_idx = 1
        

        print("Starting to transfer images' data to COCO JSON format from Hasty JSON format...")
        for img in tqdm.tqdm(uqe_imgs):

            height = img["height"]
            width = img["width"]
            img_id = img["image_id"]

            image_subsect["file_name"] = img["image_name"]
            image_subsect["height"] = min(height,width)
            image_subsect["width"] =  min(height,width)
            image_subsect["id"] = img_idx

            images_sect.append(image_subsect.copy())

            # Loading all annotations for currently considered image
            img_segs = img['labels']
            parsing_idx = 0

            # Iterating over all distinct object labels
            for obj in img_segs:

                obj_mask = numpy.asfortranarray(numpy.zeros((height, width), dtype = numpy.uint8))
                bbox = obj['bbox']
                mask = obj['mask']

                if obj["class_name"] == "Raw Cutting" or obj["class_name"] == "Tip Cutting" or obj["class_name"] == "First Section Cutting" \
                       or obj["class_name"] == "Redundant Top End" or obj["class_name"] == "Redundant Bottom End" or obj["class_name"] == "Second Section Cutting" \
                        or obj["class_name"] == "Third Section Cutting" or obj["class_name"] == "Non-Viable Part"or obj["class_name"] == "Remains":

                    if mask is not None:

                        bbox_obj_mask = numpy.asfortranarray(
                            rle_decode(mask, (bbox[3] - bbox[1], bbox[2] - bbox[0])))
                            
                        obj_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bbox_obj_mask

                        obj_rle = pycocotools.mask.encode(obj_mask)
                        
                        #contours, hierarchy = cv2.findContours((obj_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #segmentation = []

                        #for contour in contours:
                            #contour = contour.flatten().tolist()
                            #if len(contour) > 100:
                                #segmentation.append(contour)
                        #if len(segmentation) == 0:
                            #continue
                            #segmentation.append(contour)
                        

                        #cv2.imwrite('/home/student3/Desktop/test/'+str(img_idx)+'.png',segmentation) 
                    
                        
                        obj_rle["counts"] = obj_rle["counts"].decode("utf-8")
                        obj_bbox = pycocotools.mask.toBbox(obj_rle).tolist()
                        obj_area = int(pycocotools.mask.area(obj_rle))


                        
                        if obj["class_name"]== superclass or obj["class_name"]=="Remains" :
                            annotation_subsect["area"] = obj_area
                            annotation_subsect["bbox"] = obj_bbox
                            annotation_subsect["segmentation"] = obj_rle

                    if obj["class_name"] == superclass or obj["class_name"]== "Remains" :
                        annotation_subsect["category_id"] = 1 #class_id_asgmt[obj["class_name"]]
                        annotation_subsect["id"] = obj_idx  # obj_id
                        annotation_subsect["image_id"] = img_idx  # img_id
                        annotation_subsect["iscrowd"] = 0

                        parsing_idx = parsing_idx + 1

                        annotation_subsect["parsing_id"] = parsing_idx
                        annotations_sect.append(annotation_subsect.copy())

                        obj_idx = obj_idx + 1

            img_idx = img_idx + 1

    print('...conversion of data has been completed.')

    out_file["categories"] = categories_sect
    out_file["images"] = images_sect
    out_file["annotations"] = annotations_sect

    # Write the JSON output file
    with open(out_path, 'w+') as json_file:
        json_file.write(json.dumps(out_file))

if __name__ == "__main__":
    main(sys.argv[1:])
