#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay.lua"

WRK_CMD="wrk"

HOURLY_TRAFFIC=(
11062 12752 10917 10816 13707 15441 
17488 21007 20893 10000 10000 10000 
10000 10000 10000 10000 10000 10000 
10000 10000 10000 10000 10000 10000
)

NORMAL_BASE_CONN=100
MAX_TRAFFIC=21007
echo "=========================================================="
echo "🚀 [SCIDB 08/06] 하이브리드 트래픽 재연 시작"
echo "   - 특징: 높은 연결 실패율(62%) & 오후 트래픽 증가"
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
echo "🏁 08/06 재연 종료. log.txt를 확인하세요."