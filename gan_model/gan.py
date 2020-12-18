from options.test_options import TestOptions
from models import create_model
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

opt = TestOptions()

opt.aspect_ratio=1.0
opt.batch_size=1
opt.checkpoints_dir='./gan_model/checkpoints'
opt.crop_size=256
opt.dataroot='customdata'###################################
opt.dataset_mode='single' 
opt.direction='AtoB'
opt.display_id=-1
opt.display_winsize=256
opt.epoch='latest'
opt.eval=False
opt.gpu_ids=[0]
opt.init_gain=0.02
opt.init_type='normal'
opt.input_nc=3
opt.isTrain=False
opt.load_iter=0
opt.load_size=256
opt.max_dataset_size=np.inf
opt.model='test'
opt.model_suffix=''
opt.n_layers_D=3
opt.name='style_vangogh_pretrained'
opt.ndf=64
opt.netD='basic'
opt.netG='resnet_9blocks'
opt.ngf=64
opt.no_dropout=True
opt.no_flip=True
opt.norm='instance'
opt.num_test=50
opt.num_threads=0
opt.output_nc=3
opt.phase='test'
opt.preprocess='resize_and_crop'
opt.results_dir='./results/' ############################
opt.serial_batches=True
opt.suffix=''
opt.verbose=False


model = create_model(opt)
model.setup(opt)   
model.eval()

def GAN(input_image, max_size=None, shape=None):

    
    if max_size is None:
        max_size = 128

    if max(input_image.size) > max_size:
        size = max_size
    else:
        size = max(input_image.size)

    if shape is not None:
        size = shape

    resizer = transforms.Resize(size)
    
    input_image = resizer(input_image)

    input_tensor = (np.asarray(input_image, dtype='f'))/127.5-1
    input_tensor = torch.FloatTensor(input_tensor)
    input_tensor = input_tensor.T.unsqueeze(0)
    
    model.set_input({'A':input_tensor,'A_paths':'placeholder'})
    model.test()

    output_tensor = model.get_current_visuals()['fake']
    output_tensor = np.asarray(output_tensor.squeeze(0).T.cpu())
    output_tensor -= output_tensor.min()
    output_tensor /= output_tensor.max()/255
    output_tensor = np.uint8(output_tensor)
    output_image = Image.fromarray(output_tensor)

    return output_image