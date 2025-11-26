#!/bin/bash

# 로그 파일 초기화 (기존 파일 있으면 지움)
rm -f log.txt

echo "Monitoring started... Saving to log.txt"

# 1초마다 시스템 상태 수집 (무한 반복)
while true; do 
    echo "----------- $(date) -----------" | tee -a log.txt
    echo "--- ss -s ---" | tee -a log.txt
    ss -s | tee -a log.txt
    echo "--- iostat -x 1 1 ---" | tee -a log.txt
    iostat -x 1 1 | tee -a log.txt
    echo "" | tee -a log.txt
    sleep 1
done
