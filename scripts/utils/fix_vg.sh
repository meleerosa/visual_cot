#!/bin/bash

echo "📂 Fixing VG_100K folder structure..."
cd visual_genome/VG_100K/VG_100K
find . -type f -name "*.jpg" -print0 | xargs -0 -I{} mv {} ../
cd ../../../..
rmdir visual_genome/VG_100K/VG_100K

echo "📂 Fixing VG_100K_2 folder structure..."
cd visual_genome/VG_100K_2/VG_100K_2
find . -type f -name "*.jpg" -print0 | xargs -0 -I{} mv {} ../
cd ../../../..
rmdir visual_genome/VG_100K_2/VG_100K_2

echo "✅ Folder structure fix completed."