rootpath=$HOME/VisualSearch
# rootpath=VisualSearch
overwrite=0

# feature_name=mean_resnext101_resnet152
# collections=tgif-msrvtt10k
# feature_dim=4096
feature_name=$1
feature_dim=$2
collections=$3

# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 collection featnames [rootpath]"
#     exit
# fi

# if [ "$#" -gt 2 ]; then
#     rootpath=$3# fi



# bash do_combine_feature.sh $feature_name $feature_dim $collections 

python combine_features.py $feature_name $feature_dim $collections $rootpath  --overwrite $overwrite