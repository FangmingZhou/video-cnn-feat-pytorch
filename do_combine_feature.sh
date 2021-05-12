# rootpath=$HOME/VisualSearch
rootpath=VisualSearch
overwrite=0

feature_name=f1
collections=c1-c2-c3
feature_dim=20

# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 collection featnames [rootpath]"
#     exit
# fi

# if [ "$#" -gt 2 ]; then
#     rootpath=$3
# fi



python combine_features.py $feature_name $feature_dim $collections $rootpath  --overwrite $overwrite