REGION_NAME="JerusalemSmall"
OUTPUT_DIR=$HOME/tmp/cabby_run/$REGION_NAME
MAP_DIR=$OUTPUT_DIR/map

#OUTPUT_DIR_MODEL=$OUTPUT_DIR/cabby_run/manhattan
#OUTPUT_DIR_MODEL_RVS=$OUTPUT_DIR_MODEL/rvs
#OUTPUT_DIR_MODEL_RVS_FIXED_4=$OUTPUT_DIR_MODEL/rvs/fixed_4
#OUTPUT_DIR_MODEL_RVS_FIXED_5=$OUTPUT_DIR_MODEL/rvs/fixed_5
#
#OUTPUT_DIR_MODEL_HUMAN=$OUTPUT_DIR_MODEL/human
#


echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"

#rm -rf $OUTPUT_DIR
#mkdir -p $OUTPUT_DIR
#mkdir -p $MAP_DIR

#bazel-bin/cabby/geo/map_processing/map_processor --region $REGION_NAME --min_s2_level 18 --directory $OUTPUT_DIR

#bazel-bin/cabby/geo/sample_poi --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR --path $MAP_DIR/JerusalemSmall.gpkg --n_samples 4

#echo "****************************************"
#echo "*                 RVS                  *"
#echo "****************************************"
#bazel-bin/cabby/rvs/generate_rvs --rvs_data_path $MAP_DIR/Haifa_geo_paths.gpkg --save_instruction_dir $OUTPUT_DIR

#
#echo "****************************************"
#echo "*                 graph embeddings     *"
#echo "****************************************"
#
#GRAPH_EMBEDDING_PATH=$MAP_DIR/graph_embedding.pth
#bazel-bin/cabby/data/metagraph/create_graph_embedding  --region $REGION_NAME --dimensions 224 --s2_level 15 --s2_node_levels 15 --base_osm_map_filepath $MAP_DIR --save_embedding_path $GRAPH_EMBEDDING_PATH --num_walks 2 --walk_length 2

echo "****************************************"
echo "*                 models               *"
echo "****************************************"
mkdir -p $OUTPUT_DIR_MODEL
mkdir -p $OUTPUT_DIR_MODEL_RVS
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_4
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_5
mkdir -p $OUTPUT_DIR_MODEL_HUMAN


#echo "*                 Classification-Bert  - HUMAN DATA             *"
cabby/model/text/model_trainer.py  --data_dir /Users/itaimondshine/Desktop/Study/pasten/cabby/cabby/cabby/model/text/dataSamples/girit   --dataset_dir /Users/itaimondshine/Desktop/Study/pasten/cabby/cabby/cabby/model/text/dataset_dir --region 'Tel Aviv' --s2_level 16 text/dataset_dir --num_epochs 1 --train_batch_size 100 --task human --model  Classification-Bert

# Dual encoder
#bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model Dual-Encoder-Bert