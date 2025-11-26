#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_0805.lua"
WRK_CMD="wrk" 

HOURLY_TRAFFIC=(
21159 21853 22226 18822 18258 18121 
14397 13153 11476 11352 12424 12950 
11003 13904 15135 17356 21922 23957 
22466 21295 22265 21158 22656 21608
)

MAX_TRAFFIC=23957

NORMAL_MAX_CONNS=100

echo "=========================================================="
echo "🚀 [SCIDB 08/05] 하이브리드 트래픽 재연 시작"
echo "   - 특징: 역대 최대(5GB) 업로드 & 새벽 고트래픽"
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
        if [ $hour -lt 17 ]; then
            TARGET_CONN=500
        elif [ $hour -lt 21 ]; then
            TARGET_CONN=1000  
        else
            TARGET_CONN=700  
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
echo "🏁 08/05 하이브리드 재연 종료. log.txt를 확인하세요."