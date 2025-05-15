#!/bin/bash

# 디렉토리 준비
REFCOCO_DIR="data/refcoco"
mkdir -p $REFCOCO_DIR
cd $REFCOCO_DIR

echo "📦 Cloning refer GitHub repository..."
git clone https://github.com/lichengunc/refer.git temp_refer

echo "📄 Copying RefCOCO annotations..."
mkdir -p annotations
cp -r temp_refer/data/* ./annotations/

echo "🔗 Linking COCO train2014 images..."
# COCO 이미지가 여기에 있어야 함: data/coco/train2014
ln -sf ../../coco/train2014 ./images

echo "🧹 Cleaning up temporary repo..."
rm -rf temp_refer

echo "✅ RefCOCO, RefCOCO+, RefCOCOg annotations downloaded and organized!"
echo "   - annotations/: refs(unc/umd/google).p, images.json, instances.json"
echo "   - images/: linked to COCO train2014"
