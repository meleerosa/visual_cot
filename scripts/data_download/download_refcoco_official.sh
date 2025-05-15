#!/bin/bash

set -e

BASE_DIR="data/refcoco_official"
mkdir -p $BASE_DIR
cd $BASE_DIR

echo "📁 Making directories..."
mkdir -p images

##########################################
# 1. Annotation 다운로드 및 압축 해제
##########################################

function download_and_unzip {
    name=$1
    url=$2
    echo "⬇️  Downloading $name..."
    wget -nc $url -O ${name}.zip
    unzip -n ${name}.zip -d $name
    rm ${name}.zip
}

download_and_unzip refclef http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
download_and_unzip refcoco http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
download_and_unzip refcoco+ http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
download_and_unzip refcocog http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

##########################################
# 2. 이미지 폴더 구조 설명 출력
##########################################

echo ""
echo "📌 이미지 디렉토리는 아래처럼 구성해야 합니다:"
echo "  images/"
echo "  ├── mscoco/         ← COCO train2014 이미지 복사 또는 링크"
echo "  └── saiaprtc12/     ← 별도로 다운 필요"
echo ""
echo "🔗 COCO train2014 이미지 예시 링크:"
echo "    http://images.cocodataset.org/zips/train2014.zip"
echo ""
echo "🔗 saiaprtc12 이미지 (subset) 링크:"
echo "    http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip"

echo ""
echo "✅ All RefCOCO annotations downloaded and structured at: $BASE_DIR"
