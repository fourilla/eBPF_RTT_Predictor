#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay_0801.lua"

WRK_CMD="wrk"

HOURLY_TRAFFIC=(
20759 20896 21330 18995 19936 18230 
15564 13527 11425 10861 10700 12722 
10669 13047 16045 17320 17138 19276 
18236 17542 17806 18704 21593 19048
)

NORMAL_BASE_CONN=100
MAX_TRAFFIC=21593

echo "=========================================================="
echo "🚀 [SCIDB 08/01] "
echo "   - 00~11시: 실제 비율 기반 저부하 (Normal)"
echo "   - 12~23시: 강제 고부하 주입 (Stress)"
echo "=========================================================="

ulimit -n 65535

for hour in {0..23}; do
    echo ""
    
    if [ $hour -lt 12 ]; then
        # [전반전: Normal] 실제 데이터 비율대로 '연결 수'를 조절
        CURRENT_DATA=${HOURLY_TRAFFIC[$hour]}
        
        # 비율 계산: (현재트래픽 / 최대트래픽) * 50개
        TARGET_CONN=$(( ($CURRENT_DATA * $NORMAL_BASE_CONN) / $MAX_TRAFFIC ))
        if [ $TARGET_CONN -lt 5 ]; then TARGET_CONN=5; fi
        
        THREADS=2
        MODE="[Normal:실제비율]"
        
    else
        THREADS=4
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
echo "🏁 08/01 재연 종료. log.txt를 확인하세요."