'''
feature.bin
id.txt
shape.txt  
'''
import os
import sys
import numpy as np
from optparse import OptionParser

from txt2bin import checkToSkip

def process(feature_name:str, feature_dim:int, collections:list, root_path:str, overwrite:int)->None:
    # collection_list = collections
    # feature_name = 'f1'
    # root_path = './VisualSearch'
    dtype = np.float32
    # offset = np.float32(1).nbytes

    res_collection_name = '-'.join(collections)
    res_feature_dir = os.path.join(root_path, res_collection_name,\
        'FeatureData', feature_name)

    if os.path.isdir(res_feature_dir) is False:
        os.makedirs(res_feature_dir)

    res_id_file = os.path.join(res_feature_dir, 'id.txt')
    res_bin_file = os.path.join(res_feature_dir, 'feature.bin')
    res_id_list = []

    if checkToSkip(res_bin_file, overwrite):
        return 0

    fw = open(res_bin_file, 'wb')

    for collection in collections:
        feature_dir = os.path.join(root_path, collection, 'FeatureData', feature_name)
        bin_file = os.path.join(feature_dir, 'feature.bin')
        id_file = os.path.join(feature_dir, 'id.txt')
        shape_file = os.path.join(feature_dir, 'shape.txt')

        shape = list(map(int, open(shape_file, 'r').read().strip().split()))
        

        id_list = open(id_file, 'r').read().strip().split()
        feature_num = len(id_list)

        assert(shape[0] == feature_num), "Mismatch: %d %d" % (shape[0], feature_num)
        assert(shape[1] == feature_dim), "Mismatch: %d %d" % (shape[1], feature_dim)

        fr = open(bin_file, 'rb')

        # 将当前feature写入目标bin文件中
        for i in range(feature_num):
            vec = np.fromfile(fr, dtype=dtype, count=feature_dim)
            # print(vec)
            vec.tofile(fw)
    
        fr.close()

        res_id_list += open(id_file,'r').read().strip().split()

    fw.close()

    # 写目标id文件
    # 注意，如果bin文件没有写入完毕，id文件是不会生成的，可以以id文件是否生成作为是否合并完成的标志。
    fw = open(res_id_file,'w')
    fw.write(' '.join(res_id_list))
    fw.close()

    fw = open(os.path.join(res_feature_dir, 'shape.txt'), 'w')
    fw.write('%d %d' % (len(res_id_list), feature_dim)) 


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    
    parser = OptionParser(usage="""usage: %prog [options] feature_name feature_dim collections root_path \n \
        Tips: use '-' to concat the target collections, for example: collections=msrvtt10k-tgif-vatex""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    (options, args) = parser.parse_args(argv)
    if len(args) != 4:
        parser.print_help()
        return 1

    feature_name = args[0]
    feature_dim = int(args[1])
    collections = args[2].split('-')
    root_path = args[3]

    
    assert(len(collections) > 1), '%s dont have more then one collection' % collections
    assert(type(feature_dim) == int and feature_dim > 0), 'Feature dimension should be a positive int number'
    
    return process(feature_name, feature_dim, collections, root_path, options.overwrite)


if __name__ == "__main__":
    sys.exit(main())