#!/bin/bash

# ssh 자동 로그인에 필요한 정보
USER="wlaud"
HOST="165.132.146.91"
PASSWORD="wlaud"

# sshpass 설치 필요: sudo apt install sshpass
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ${USER}@${HOST} '
  echo "✅ 접속 완료. GPU 사용량과 로그 모니터링 시작..."
  while true; do
    clear
    echo "========== [GPU 사용량 - nvidia-smi] =========="
    nvtop || echo "⚠️ nvidia-smi 실행 실패"
    echo
    echo "========== [train.log 마지막 줄] =========="
    tail -n 1 /root/project/train.log 2>/dev/null || echo "⚠️ 로그 파일 없음"
    sleep 1
  done
'