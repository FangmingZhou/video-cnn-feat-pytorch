import os
import sys
import json
import time
import torch
import logging
from torch.utils.data import DataLoader

from constant import *
from utils.generic_utils import Progbar
from data_provider import ImageDataset


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_name(model_name, layer, oversample):
    feat = model_name
    return '%s,%s,os' % (feat,layer) if oversample else '%s,%s' % (feat, layer)


def get_model(model_dir, model_name):
    if os.path.exists(os.path.join(model_dir, model_name)):
        origin_model = torch.hub.load(model_dir, model_name, source='local')
    else:
        origin_model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        
    model = torch.nn.Sequential(*list(origin_model.children())[:-1])
    model.eval()
    if torch.cuda.is_available():
        model.to(device)
    print(model)
    return model


def process(options, collection):
    rootpath = options.rootpath
    oversample = options.oversample

    model_dir = os.path.join(rootpath, options.model_dir)
    model_name = options.model_name

    layer = 'avgpool'
    batch_size = options.batch_size
    feat_name = get_feature_name(model_name, layer, oversample)

    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feat_name)
    id_file = os.path.join(feat_dir, 'id.txt')
    feat_file = os.path.join(feat_dir, 'id.feature.txt')
    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
    
    if options.split != "":
        id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt.split', options.split)
        feat_file += options.split
        print('id_path_file:%s \nfeature_file:%s'%(id_path_file, feat_file))

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)


    for x in [id_file, feat_file]:
        if os.path.exists(x):
            if not options.overwrite:
                logger.info('%s exists. skip', x)
                return 0
            else:
                logger.info('%s exists. overwrite', x)

    
    
    model = get_model(model_dir, model_name)
   
    dataset = ImageDataset(id_path_file, oversample=oversample)
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    logger.info('%d images', len(dataset))

    fw = open(feat_file, 'w')
    progbar = Progbar(len(dataset))  
    start_time = time.time()
    for image_ids, image_tensor in dataloder:
        
        batch_size = len(image_ids)
        if torch.cuda.is_available():
            image_tensor = image_tensor.to(device)

        if oversample:
            _, ncrops, c, h, w = image_tensor.size()
            image_tensor = image_tensor.view(-1,c,h,w)

        with torch.no_grad():
            output = model(image_tensor)

        if oversample:
            output = output.view(batch_size, ncrops, -1).mean(1)
        else:
            output = output.view(batch_size, -1)

        target_feature = output.cpu().numpy()
        for i, image_id in enumerate(image_ids):
            fw.write('%s %s\n' % (image_id, ' '.join( ['%g'%x for x in target_feature[i] ])))
            
            
        progbar.add(batch_size)
    elapsed_time = time.time() - start_time
    logger.info('total running time %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    fw.close()

    
        #  >>> input, target = batch # input is a 5d tensor, target is 2d
        #  >>> bs, ncrops, c, h, w = input.size()
        #  >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
        #  >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--gpu", default=0, type="int", help="gpu id (default: 0)")
    parser.add_option("--oversample", default=0, type="int", help="oversample (default: 0)")
    parser.add_option("--model_dir",default=DEFAULT_MODEL_DIR, type="string")
    parser.add_option("--model_name",default=DEFAULT_MODEL_NAME, type="string")
    parser.add_option("--batch_size",default=1, type="int")
    parser.add_option("--split",default='', type="string", help="deal one split part of entire collection")
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    print(json.dumps(vars(options), indent = 2))
    
    return process(options, args[0])


if __name__ == '__main__':
    sys.exit(main())

