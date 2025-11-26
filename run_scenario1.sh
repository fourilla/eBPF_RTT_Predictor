#!/bin/bash

TARGET="http://localhost:80"
LUA_SCRIPT="./scenario_replay1.lua"
WRK_CMD="wrk"

LOAD_PATTERN=(50 200 500 700 1000 3000 5000 10000 50000 70000 100000 70000 50000 10000 5000 3000 1000 700 500 200 50)

THREADS=8
DURATION="1m"

echo "=========================================================="
echo "🚀 Traffic Replay Started"
echo "   Target: $TARGET"
echo "=========================================================="

ulimit -n 65535

for i in "${!LOAD_PATTERN[@]}"; do
    CONNS=${LOAD_PATTERN[$i]}
    STEP=$((i+1))
    
    echo ""
    echo "Step $STEP: Connections $CONNS"
    
    $WRK_CMD -t$THREADS -c$CONNS -d$DURATION -s $LUA_SCRIPT $TARGET
    
    echo "Step $STEP completed."
done

echo ""
echo "🏁 Scenario Finished. Check log.txt."