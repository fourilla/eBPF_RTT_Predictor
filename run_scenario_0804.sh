#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_0804.lua"

WRK_CMD="wrk"

HOURLY_TRAFFIC=(
22601 23552 22443 21515 18096 18527
15855 13596 12062 11448 10636 11598
11458 13758 15040 17568 21308 22895
22209 22111 23714 27767 24889 21750
)

NORMAL_BASE_CONN=100
MAX_TRAFFIC=27767

echo "=========================================================="
echo "🚀 [SCIDB 08/03] "
echo "   - 00:00 ~ 11:00 : Real Pattern Replay (평시 모드)"
echo "   - 12:00 ~ 23:00 : Artificial Stress Test (폭주 모드)"
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
    
    echo "✅ $hour시 구간 완료."
    sleep 2
done

echo ""
echo "🏁 08/04 재연 종료. log.txt를 확인하세요."