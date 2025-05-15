echo "📦 시작: Visual Genome 이미지 디렉토리 정리..."

# 중첩된 VG_100K 제거
rm -rf data/visual_genome/images/VG_100K_2/VG_100K 2>/dev/null || true
rm -rf data/gqa/images/VG_100K_2/VG_100K 2>/dev/null || true

# 이미지 루트에서 정리
mkdir -p data/visual_genome/images

# VG_100K 정위치
if [ -d data/visual_genome/VG_100K ]; then
    mv data/visual_genome/VG_100K data/visual_genome/images/VG_100K
fi

if [ -d data/visual_genome/VG_100K_2 ]; then
    mv data/visual_genome/VG_100K_2 data/visual_genome/images/VG_100K_2
fi

# gqa 이미지 심볼릭 링크로 통일
rm -rf data/gqa/images
ln -s ../visual_genome/images data/gqa/images

# lvis 이미지 링크 통일
rm -rf data/lvis/images
ln -s ../coco/train2017 data/lvis/images

echo "✅ 정리 완료! 이미지 디렉토리 중첩 제거 + 링크 구성 완료"
