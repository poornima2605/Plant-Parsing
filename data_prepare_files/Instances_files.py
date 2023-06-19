import json
import PIL
import numpy
import matplotlib.image
import sys, getopt, os, shutil
from PIL import ImageChops, Image
import numpy  as np



def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts = starts - 1
    ends = starts + lengths
    img = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def main(argv):
    annotation_file = ''
    input_folder = ''
    output_folder = ''

    try:
        opts, args = getopt.getopt(argv, "ha:i:o:r:", ["annotation_file=", "input_folder", "output_folder", "repetitions"])

        # If input file and output folder are not added
        if len(opts)< 3:
            print('usage: python binary_mask_for_every_Hasty_object.py -a <annotation_file> -i <input_folder> -o <output_folder> -r <repetitions>')
            sys.exit(2)

    except getopt.GetoptError:
        print('python binary_mask_for_every_Hasty_object.py -a <annotation_file> -i <input_folder> -o <output_folder> -r <repetitions>')
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            print('python binary_mask_for_every_Hasty_object.py -a <annotation_file> -i <input_folder> -o <output_folder> -r <repetitions>')
            sys.exit()

        elif opt in ("-a", "--annotation_file"):
            annotation_file = arg

        elif opt in ("-i", "--input_folder"):
            input_folder = arg

        elif opt in ("-o", "--output_folder"):
            output_folder = arg

        elif opt in ("-r", "--repetitions"):
            reps = int(arg)

    if 'reps' not in vars():
        reps = 1

    print('Opening json file: {}'.format(annotation_file))

    with open(annotation_file) as file:

        if not input_folder[-1] == '/':
            input_folder = input_folder + "/"

        if not output_folder[-1] == '/':
            output_folder = output_folder + "/"

        if not os.path.exists(input_folder):
            print('Input folder could not be located.')
            sys.exit(3)

        if not os.path.exists(output_folder): # Making output directory
            print('The specified output folder does not exist yet. It will be created.')
            os.makedirs(output_folder)

        json_file = json.load(file)  # Reading json file

        # Accessing the 'images' section of the JSON file
        imgs = json_file['images']

        # Keeping only the unique objects in the list of dictionaries
        uqe_imgs = {x['image_id']:x for x in imgs}.values()

        # Keeping the user up to date on found number of unique images and distinct object labels
        print('Number of unique images in file: {}'.format(len(uqe_imgs)))
        print('Number of distinct objects in file: {}'.format(sum([len(x['labels']) for x in uqe_imgs])))

        print('Starting to create binary image masks...')
        for img in uqe_imgs:
            

            # Concatening directory and file name for obtaining the image's absolute path
            input_img_path = input_folder + img['image_name']
            output_img_path1 = output_folder +'Instances/' + img['image_name']
            output_img_path2 = output_folder + 'Instance_ids/' + img['image_name']
            #img_idx=1
            
            if not os.path.exists(output_img_path1): # Making output directory
               print('The specified output folder does not exist yet. It will be created.')
               os.makedirs(output_folder)
               
            if not os.path.exists(output_img_path2): # Making output directory
               print('The specified output folder does not exist yet. It will be created.')
               os.makedirs(output_folder)
            
            #file1= open (output_img_path+".txt","w")
            img_idx=1
            count=[]

            if os.path.exists(input_img_path):
                

                # Loading all annotations for currently considered image
                img_segs = img['labels']
                #count1,count2,count3,count4,count5,count6,count7,count8 =0,0,0,0,0,0,0,0
                

                if len(img_segs) > 0:
                    obj_masks1 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8) 
                    obj_masks2 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)
                    obj_masks3 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8) 
                    obj_masks4 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)

                    # Iterating over all distinct object labels
                    obj_idx1 = 1
                    obj_idx2 = 1
                    for obj in img_segs:
                        

                        if obj['class_name'] == 'Tip Cutting'or obj['class_name'] == 'First Section Cutting' or obj['class_name'] == 'Redundant Bottom End' or obj['class_name'] == 'Second Section Cutting' or obj['class_name'] == 'Third Section Cutting'or obj['class_name'] == 'Redundant Top End'or obj['class_name'] == 'Non-Viable Part':
                            
                            if obj['class_name'] == 'Tip Cutting':         
                                value=2
                                value1=[168,196,84]
                                
                            elif obj['class_name'] == 'First Section Cutting':
                                value=4
                                value1=[56,140,224]
                                
                            elif obj['class_name'] == 'Redundant Bottom End':
                                value=7 
                                value1=[196,112,56]
                                
                            elif obj['class_name'] == 'Second Section Cutting':
                                value=5
                                value1=[112,168,224]
                                
                            elif obj['class_name'] == 'Third Section Cutting':
                                value=6
                                value1=[224,224,84]
                                
                            elif obj['class_name'] == 'Redundant Top End':
                                value=3
                                value1=[112,56,112]
                                
                            elif obj['class_name'] == 'Non-Viable Part':
                                value=1
                                value1=[0,112,196]
                                
                            else:
                                value=0
                                value1=[0,0,0]
                            count.append(value)   
                                
                            obj_mask1 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8)
                            obj_mask2 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)
                            
                            bbox = obj['bbox']
                            bbox_obj_mask = rle_decode(obj['mask'], (bbox[3] - bbox[1], bbox[2] - bbox[0]))
                            
                            #To be solved for rGB
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],0] = bbox_obj_mask  * value1[0] 
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],1] = bbox_obj_mask  * value1[1]
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],2] = bbox_obj_mask  * value1[2]
                            
                            obj_mask2[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bbox_obj_mask *value
                            

                            obj_masks1= numpy.add( obj_masks1, obj_mask1)
                            obj_masks2= numpy.add( obj_masks2, obj_mask2)
                            
                                                          
                            obj_idx1 = obj_idx1 + 1
                            
                        elif obj['class_name'] == 'Raw Cutting' : 
                            
                            obj_mask3 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8)
                            obj_mask4 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)
                            
                            bbox = obj['bbox']
                            bbox_obj_mask = rle_decode(obj['mask'], (bbox[3] - bbox[1], bbox[2] - bbox[0]))
                            
                            temp =numpy.array([int(i) for i in list('{0:03b}'.format(obj_idx2))])
      
                            
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],0] = bbox_obj_mask  *temp[2]
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],1] = bbox_obj_mask  *temp[1]
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],2] = bbox_obj_mask  *temp[0]
                            
                            obj_mask4[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bbox_obj_mask * (obj_idx2-1) *20
                            
                            obj_masks3= numpy.add( obj_masks3, obj_mask3)
                            obj_masks4= numpy.add( obj_masks4, obj_mask4)
                            
                            obj_idx2 = obj_idx2 + 1 
                            
                        elif obj['class_name'] == 'Remains' :
                         
                            value=8
                            value1=[224,84,84]
                            obj_mask1 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8)
                            obj_mask2 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)
                            obj_mask3 = numpy.zeros((img["height"], img["width"],3), dtype = numpy.uint8)
                            obj_mask4 = numpy.zeros((img["height"], img["width"]), dtype = numpy.uint8)
                            
                            bbox = obj['bbox']
                            bbox_obj_mask = rle_decode(obj['mask'], (bbox[3] - bbox[1], bbox[2] - bbox[0]))
                            
                            #To be solved for rGB
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],0] = bbox_obj_mask  * value1[0] 
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],1] = bbox_obj_mask  * value1[1]
                            obj_mask1[bbox[1]:bbox[3], bbox[0]:bbox[2],2] = bbox_obj_mask  * value1[2]
                            
                            obj_mask2[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bbox_obj_mask *value
                            obj_masks1= numpy.add( obj_masks1, obj_mask1)
                            obj_masks2= numpy.add( obj_masks2, obj_mask2)                                                        
                             
                            temp =numpy.array([int(i) for i in list('{0:03b}'.format(obj_idx2))])   
                            
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],0] = bbox_obj_mask  *temp[2]
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],1] = bbox_obj_mask  *temp[1]
                            obj_mask3[bbox[1]:bbox[3], bbox[0]:bbox[2],2] = bbox_obj_mask  *temp[0]
                            
                            obj_mask4[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bbox_obj_mask * (obj_idx2-1) *20
                            
                            obj_masks3= numpy.add( obj_masks3, obj_mask3)
                            obj_masks4= numpy.add( obj_masks4, obj_mask4)
                            
                            obj_idx2 = obj_idx2 + 1 
                            obj_idx1 = obj_idx1 + 1
                            
                            
                            
                        obj_masks = numpy.empty((img["height"], img["width"],3), dtype = numpy.uint8)
                        obj_masks[:,:,0]= obj_masks1[:,:,0] * obj_masks3[:,:,0]
                        obj_masks[:,:,1]= obj_masks1[:,:,1] * obj_masks3[:,:,1]
                        obj_masks[:,:,2]= obj_masks1[:,:,2] * obj_masks3[:,:,2]
                        
                        obj_masks5 = numpy.empty((img["height"], img["width"],3), dtype = numpy.uint8)
                        obj_masks5 = obj_masks2 + obj_masks4
                        
                        new_image1=Image.fromarray(obj_masks,'RGB')
                        new_image2=Image.fromarray(obj_masks5,'L')
                        
                        new_image1.save(output_img_path1)
                        new_image2.save(output_img_path2)

# =============================================================================
#                 # Writting val id files
#                 count_sort=np.sort(count)
#                 (unique,counts) = np.unique (count_sort,return_counts=True)
#                 abc =np.zeros([np.sum(counts),2],dtype=int)
#                 j,x=0,0
#                 for p in range(max(unique)):
#                     q=p
#                     for index1,value in enumerate(counts):
#                         if ((x < np.sum(counts)) and (q < np.sum(counts[:index1+1]))):
#                             abc[x,0]=count_sort[q]
#                             abc[x,1]=j+1
#                             q+=value
#                             x+=1
#                     j+=1
#                         
#                 for xy in (abc):
#                     value_1= 20 *(xy[1] -1) + xy[0] 
#                     obj_idx_value = str(value_1) + " " +str( xy[0] ) +" " +str ( xy[1] ) + "\n"   
#                     file1.write(obj_idx_value)
#                 file1.write(str(abc))            
#                         #shutil.copyfile(input_img_path, obj_input_img_path)
#                 if len(img_segs) > 0:
#                         
#                         #img_idx=img_idx+1
#                 file1.close()
# =============================================================================

                os.remove(input_img_path)
            else:
                print(input_img_path, "does not exist!")

    print('...binary image mask creation has been completed.')

if __name__ == "__main__":
    main(sys.argv[1:])
