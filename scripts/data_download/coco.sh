#!/bin/bash

# Working directory: project/scripts/data_download

echo ">> Downloading MS COCO 2014 images..."

# 다운로드 위치 설정
COCO_DIR="../../data/coco"
mkdir -p ${COCO_DIR}
cd ${COCO_DIR}

# Train 이미지 (~13GB)
if [ ! -d "train2014" ]; then
  echo ">> Downloading train2014..."
  wget http://images.cocodataset.org/zips/train2014.zip
  unzip train2014.zip
  rm train2014.zip
else
  echo ">> train2014 already exists. Skipping..."
fi

# Val 이미지 (~6GB)
if [ ! -d "val2014" ]; then
  echo ">> Downloading val2014..."
  wget http://images.cocodataset.org/zips/val2014.zip
  unzip val2014.zip
  rm val2014.zip
else
  echo ">> val2014 already exists. Skipping..."
fi

echo "✅ COCO 2014 image download complete!"
