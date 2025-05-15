#!/bin/bash

BASE_DIR="data"
mkdir -p $BASE_DIR

##################################
# 1. LVIS + COCO 2017
##################################

echo "📦 Downloading LVIS annotations + COCO 2017 train images..."

LVIS_DIR="$BASE_DIR/lvis"
COCO_DIR="$BASE_DIR/coco"

mkdir -p $LVIS_DIR/annotations
mkdir -p $COCO_DIR/train2017

# LVIS Annotations
wget -nc -P $LVIS_DIR/annotations https://s3.amazonaws.com/oss.lvis.io/annotations/lvis_v1_train.json
wget -nc -P $LVIS_DIR/annotations https://s3.amazonaws.com/oss.lvis.io/annotations/lvis_v1_val.json
wget -nc -P $LVIS_DIR/annotations https://s3.amazonaws.com/oss.lvis.io/annotations/lvis_v1_image_info.json

# COCO 2017 train images (~19GB)
cd $COCO_DIR
if [ ! -d "train2017" ] || [ -z "$(ls -A train2017)" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip
    rm train2017.zip
fi
cd -

# Link COCO train2017 images to LVIS
ln -sf ../../coco/train2017 $LVIS_DIR/images

echo "✅ LVIS + COCO 다운로드 완료"


##################################
# 2. Visual Genome
##################################

echo "📦 Downloading Visual Genome..."

VG_DIR="$BASE_DIR/visual_genome"
mkdir -p $VG_DIR/images

cd $VG_DIR

# 핵심 JSON 어노테이션
wget -nc https://cs.stanford.edu/people/rak248/VG/object_relationships.json -O relationships.json
wget -nc https://cs.stanford.edu/people/rak248/VG/objects.json -O objects.json
wget -nc https://cs.stanford.edu/people/rak248/VG/region_descriptions.json -O region_descriptions.json

# 이미지 (~100GB)
if [ ! -d "images" ] || [ -z "$(ls -A images)" ]; then
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    unzip images.zip -d images
    rm images.zip
fi

cd -

echo "✅ Visual Genome 다운로드 완료"


##################################
# 3. GQA (annotation만)
##################################

echo "📦 Downloading GQA..."

GQA_DIR="$BASE_DIR/gqa"
mkdir -p $GQA_DIR/images
cd $GQA_DIR

# 이미지 (Visual Genome 공유)
ln -sf ../../visual_genome/images ./images

# 질문 + sceneGraphs (~6GB)
wget -nc https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip -n questions1.2.zip -d questions
rm questions1.2.zip

wget -nc https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
unzip -n sceneGraphs.zip -d .
rm sceneGraphs.zip

cd -

echo "✅ GQA 다운로드 완료"
echo "🎉 All datasets are downloaded and organized!"
