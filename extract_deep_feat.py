import os
import sys
import json
import time
import torch
import logging
from torch.utils.data import DataLoader

from constant import *
from utils.generic_utils import Progbar
from feat_os import extract_feature
from data_provider import ImageDataset


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feat_name(model_name, layer, oversample):
    feat = model_name
    return '%s,%s,os' % (feat,layer) if oversample else '%s,%s' % (feat, layer)


def extract_pytorch_feat(model, layer, image_id, image_path, oversample):

    image_id_list, features = extract_feature(model, layer, 1, [image_id], [image_path], oversample=oversample)
    # out_feature = features[0].view(-1).cpu().numpy()

    return image_id_list[0], features[0]


def get_model(model_dir, model_name):
    model = torch.hub.load(model_dir, model_name, source='local')

    print(model)
    return model


def process(options, collection):
    rootpath = options.rootpath
    oversample = options.oversample

    model_dir = os.path.join(rootpath, options.model_dir)
    model_name = options.model_name

    layer = 'avgpool'
    batch_size = 1 # change the batch size will get slightly different feature vectors. So stick to batch size of 1.
    feat_name = get_feat_name(model_name, layer, oversample)

    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feat_name)
    id_file = os.path.join(feat_dir, 'id.txt')
    feat_file = os.path.join(feat_dir, 'id.feature.txt')
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)


    for x in [id_file, feat_file]:
        if os.path.exists(x):
            if not options.overwrite:
                logger.info('%s exists. skip', x)
                return 0
            else:
                logger.info('%s exists. overwrite', x)

    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
    # data = list(map(str.strip, open(id_path_file).readlines()))
    # image_ids = [x.split()[0] for x in data]
    # file_names = [x.split()[1] for x in data]

    model = get_model(model_dir, model_name)
    if torch.cuda.is_available():
        model.to(device)
    
    model.eval()
    

    features = []
    def hook(module, input, output):
        features.append(output.clone().detach())
        # features = output
    handle = model.avgpool.register_forward_hook(hook)

    # output.view(-1).detach().cpu().numpy()

    
    # torch.nn.Module.register_forward_hook()
    
    dataset = ImageDataset(id_path_file)
    dataloder = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    logger.info('%d images', len(dataset))

    fw = open(feat_file, 'w')
    progbar = Progbar(len(dataset))  
    start_time = time.time()
    # import pdb; pdb.set_trace()
    for image_ids, image_tensor in dataloder:
        batch_size = len(image_ids)
        if torch.cuda.is_available():
            image_tensor = image_tensor.to(device)
        with torch.no_grad():
            model(image_tensor)
        target_feature = features[0].view(batch_size, -1).data.cpu().numpy()
        features = []
        # feature = None
        for i, image_id in enumerate(image_ids):
            fw.write('%s %s\n' % (image_id, ' '.join(['%g'%x for x in target_feature[i]])))
            pass
            
        progbar.add(batch_size)
    elapsed_time = time.time() - start_time
    logger.info('total running time %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    fw.close()

    

    # fails_id_path = []
    

    # im2path = list(zip(image_ids, file_names))
    # success = 0
    # fail = 0

    # start_time = time.time()
    # logger.info('%d images, %d done, %d to do', len(image_ids), 0, len(image_ids))
    # progbar = Progbar(len(im2path))

    # for i, (image_id, image_path) in enumerate(im2path):
    #     try:
    #         image_id, features = extract_pytorch_feat(model, layer, image_id, image_path, oversample)
    #         # import pdb; pdb.set_trace()
    #         fw.write('%s %s\n' % (image_id, ' '.join(['%g'%x for x in features])))
    #         success += 1
    #         # del features
    #     except Exception as e:
    #         fail += 1
    #         logger.error('failed to process %s', image_path)
    #         logger.info('%d success, %d fail', success, fail)
    #         fails_id_path.append((image_id, image_path))
    #     finally:
    #         progbar.add(1)

    # logger.info('%d success, %d fail', success, fail)
    # elapsed_time = time.time() - start_time
    # logger.info('total running time %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
 

    # if len(fails_id_path) > 0:
    #     fail_fw = open(os.path.join(rootpath, collection, 'feature.fails.txt'), 'w')
    #     for (imgid, impath) in fails_id_path:
    #         fail_fw.write('%s %s\n' % (imgid, impath))
    #     fail_fw.close()


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
    
  
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    print(json.dumps(vars(options), indent = 2))
    
    return process(options, args[0])


if __name__ == '__main__':
    sys.exit(main())

