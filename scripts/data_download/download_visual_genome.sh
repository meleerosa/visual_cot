#!/bin/bash

VG_DIR="data/visual_genome"
mkdir -p $VG_DIR
cd $VG_DIR

echo "📦 Downloading Visual Genome annotations..."

# 핵심 JSON 파일들
wget -nc https://cs.stanford.edu/people/rak248/VG/object_relationships.json -O relationships.json
wget -nc https://cs.stanford.edu/people/rak248/VG/objects.json -O objects.json
wget -nc https://cs.stanford.edu/people/rak248/VG/region_descriptions.json -O region_descriptions.json

echo "📦 Downloading VG_100K..."
wget -nc https://cs.stanford.edu/people/rak248/VG_100K/images.zip -O VG_100K.zip
unzip -n VG_100K.zip -d VG_100K
rm VG_100K.zip

echo "📦 Downloading VG_100K_2..."
wget -nc https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O VG_100K_2.zip
unzip -n VG_100K_2.zip -d VG_100K_2
rm VG_100K_2.zip

echo "✅ Visual Genome 다운로드 완료!"
