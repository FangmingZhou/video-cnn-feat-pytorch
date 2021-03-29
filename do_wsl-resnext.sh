rootpath=~/VisualSearch
oversample=0
overwrite=0

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

test_collection=$1

model_dir=pytorch_models/facebookresearch_WSL-Images
# model_name=resnext101_32x48d_wsl
model_name=resnext101_32x8d_wsl


raw_feat_name=${model_name},avgpool
gpu_id=2
./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_dir} ${model_name}

