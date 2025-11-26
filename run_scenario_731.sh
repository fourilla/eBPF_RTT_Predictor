#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_731.lua"

WRK_CMD="wrk"

HOURLY_TRAFFIC=(
17515 18782 16629 17785 16390 19007 
15286 12938 11222 12119 11380 11573 
11169 13954 13693 16358 17629 16018 
16955 16847 15758 16772 18847 17635
)

NORMAL_BASE_CONN=100
MAX_TRAFFIC=19007

echo "=========================================================="
echo "🚀 [SCIDB 07/31] 하이브리드 트래픽 재연 시작"
echo "   - 특징: 새벽 05시 피크 & 4GB 초대용량 업로드"
echo "=========================================================="
ulimit -n 65535

for hour in {0..23}; do
    echo ""
    
    if [ $hour -lt 12 ]; then
        # [전반전: Normal] 실제 데이터 비율대로 '연결 수'를 조절
        CURRENT_DATA=${HOURLY_TRAFFIC[$hour]}
        
        TARGET_CONN=$(( ($CURRENT_DATA * $NORMAL_BASE_CONN) / $MAX_TRAFFIC ))
        if [ $TARGET_CONN -lt 5 ]; then TARGET_CONN=5; fi
        
        THREADS=2
        MODE="[Normal:실제비율]"
        
    else
        THREADS=8
        if [ $hour -lt 16 ]; then
            TARGET_CONN=500
        elif [ $hour -lt 20 ]; then
            TARGET_CONN=700
        else
            TARGET_CONN=1000
        fi
        MODE="[Stress:강제폭주]"
    fi

    printf "🕒 %02d:00 %s 연결 수: %3d 개 \n" $hour "$MODE" $TARGET_CONN
    
    $WRK_CMD -t$THREADS -c$TARGET_CONN -d1m -s $LUA_SCRIPT $TARGET
    
    if [$hour -eq 12]; then
        echo "💡 강제 부하 시작"
    fi
    
    echo "✅ $hour시 구간 완료."
    sleep 2
done

echo ""
echo "🏁 07/31 재연 종료. log.txt를 확인하세요."