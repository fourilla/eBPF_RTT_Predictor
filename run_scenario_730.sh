#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_730.lua"
WRK_CMD="wrk" 

HOURLY_TRAFFIC=(
18816 21270 17878 18657 17653 17354 
14925 13645 12058 10000 10000 10000 
10000 10000 10000 10000 10000 10000 
20802 21626 22167 20304 20248 19902
)

MAX_TRAFFIC=22167

NORMAL_MAX_CONNS=100

echo "=========================================================="
echo "🚀 [SCIDB 07/30] 하이브리드 트래픽 재연 시작"
echo "   - 특징: 야간/새벽 집중 & 대용량 백업 업로드"
echo "=========================================================="
ulimit -n 65535

for hour in {0..23}; do
    echo ""
    
    if [ $hour -lt 12 ]; then
        CURRENT_TRAFFIC=${HOURLY_TRAFFIC[$hour]}

        TARGET_CONN=$(( ($CURRENT_TRAFFIC * $NORMAL_MAX_CONNS) / $MAX_TRAFFIC ))
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

    printf "🕒 %02d:00 %s 연결 수: %4d 개 \n" $hour "$MODE" $TARGET_CONN
    
    $WRK_CMD -t$THREADS -c$TARGET_CONN -d1m -s $LUA_SCRIPT $TARGET
    
    if [$hour -eq 12]; then
        echo "💡 강제 부하 시작"
    fi

    echo "✅ $hour시 구간 완료."
    sleep 2
done

echo ""
echo "🏁 07/30 하이브리드 재연 종료. log.txt를 확인하세요."