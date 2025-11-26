#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_0803.lua"

WRK_CMD="wrk"

HOURLY_TRAFFIC=(
19594 21801 19453 20245 19682 18920 
15301 12953 12039 10734 10417 10533 
11128 13893 16043 18204 21333 22873 
21316 22582 23379 21209 21287 21389
)

NORMAL_BASE_CONN=100
MAX_TRAFFIC=23379

echo "=========================================================="
echo "🚀 [SCIDB 08/03] "
echo "   - 특징: 20시 피크 & 대용량 업로드 부하"
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
        elif [ $hour -lt 21 ]; then
            TARGET_CONN=1000 
        else
            TARGET_CONN=700
        fi
        MODE="[Stress:강제폭주]"
    fi

    printf "🕒 %02d:00 %s 연결 수: %3d 개 \n" $hour "$MODE" $TARGET_CONN
    
    $WRK_CMD -t$THREADS -c$TARGET_CONN -d1m -s $LUA_SCRIPT $TARGET
    
    echo "✅ $hour시 구간 완료."
    sleep 2
done

echo ""
echo "🏁 08/03 재연 종료. log.txt를 확인하세요."