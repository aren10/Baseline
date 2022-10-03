import numpy as np
from models.image_clip import Image_CLIP
from PIL import Image
import os
import torch

if __name__=='__main__':
    args = {
        'model': 'vit14',
        'alpha': 0.75,
        'aggregation': 'mean',
        'n_segments': list(range(1,2,1)),
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }
    model = Image_CLIP(**args).cuda()

    root_path = '/gpfs/data/ssrinath/'
    #root_path = '../data/'

    data_path = root_path + 'toybox-13/0/'
    directories = os.listdir(data_path)
    for filename in directories:
        if filename[0:4] == 'rgba':
            img_path = data_path + filename
            im = np.array(Image.open(img_path).convert("RGB")) #im shape is (256, 256, 3)
            o_im = Image.fromarray(im).convert ('RGB')
            o_im.save(root_path + "Nesf0/"+filename)
            #heatmap
            image_clip_feature = model(im) #image_clip_feature's size is torch.Size([1, 768, 1])
            np.save(root_path + "Nesf0/"+filename[:-4]+"_image_clip_feature", image_clip_feature)
            #score = model.verify(image_clip_feature, "one, two, three and four") # score:  [0.17674114]
            #score = model.verify(image_clip_feature, "chair") # score:  [0.1314615]
            #score = model.verify(image_clip_feature, "apple, bench, sun, sky") # score:  [0.09562321]
            #image_clip_feature = torch.tensor(np.load("data/Nesf0/rgba_00094_image_clip_feature.npy"))
            #score = model.verify(image_clip_feature, "airplane, chair, rifle, boat") # score: [0.20882909]
            #print("score: ", score)
            print(filename+" saved")
