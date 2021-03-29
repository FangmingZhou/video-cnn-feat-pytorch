gpu_id=$1 
rootpath=$2
oversample=$3
overwrite=$4
raw_feat_name=$5
test_collection=$6
model_dir=$7
model_name=$8
# model_prefix=$7

if [ "$oversample" -eq 1 ]; then
    raw_feat_name=${raw_feat_name},os
fi  

BASEDIR=$(dirname "$0")
python ${BASEDIR}/generate_imagepath.py ${test_collection} --overwrite 0 --rootpath $rootpath
imglistfile=$rootpath/${test_collection}/id.imagepath.txt

if [ ! -f $imglistfile ]; then
    echo "$imglistfile does not exist"
    exit
fi


CUDA_VISIBLE_DEVICES=$gpu_id python ${BASEDIR}/extract_deep_feat.py ${test_collection}  --oversample $oversample --gpu ${gpu_id} --overwrite $overwrite --rootpath $rootpath --model_dir $model_dir --model_name $model_name

feat_dir=$rootpath/${test_collection}/FeatureData/$raw_feat_name
feat_file=$feat_dir/id.feature.txt

# exit
if [ -f ${feat_file} ]; then
    python ${BASEDIR}/txt2bin.py 0 $feat_file 0 $feat_dir --overwrite $overwrite
    # rm $feat_file
fi
