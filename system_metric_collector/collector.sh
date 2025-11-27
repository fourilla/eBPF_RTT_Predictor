#!/bin/bash
# =================================================================================
# [AI 학습용 데이터 수집기 - Anti-Lag Edition]
# =================================================================================

set -euo pipefail

# ---------------------------------------------------------------------------------
# 1. 설정 (Configuration)
# ---------------------------------------------------------------------------------
LOG_DIR="AI_TRAINING_DATA_$(date +%Y%m%d_%H%M%S)"
MAX_DURATION=1800  # 최대 30분 (시나리오 끝나면 알아서 꺼짐)

echo ">> [$(date +%T)] Master Collector (Anti-Lag Ver) 시작."
echo ">> 로그 저장 경로: $LOG_DIR"
mkdir -p "$LOG_DIR"

PIDS=() 

# CPU Throttling 확인용 경로
CPU_STAT_FILE=""
if [ -f /sys/fs/cgroup/cpu/cpu.stat ]; then CPU_STAT_FILE="/sys/fs/cgroup/cpu/cpu.stat"
elif [ -f /sys/fs/cgroup/cpu.stat ]; then CPU_STAT_FILE="/sys/fs/cgroup/cpu.stat"; fi

# ---------------------------------------------------------------------------------
# 2. 종료 처리 함수 (Cleanup)
# ---------------------------------------------------------------------------------
cleanup() {
    echo ""
    echo ">> [$(date +%T)] Cleanup: 백그라운드 수집기 종료 중..."
    if ((${#PIDS[@]} > 0)); then 
        kill "${PIDS[@]}" 2>/dev/null || true 
    fi
    echo ">> [$(date +%T)] 모든 수집 종료 완료. 데이터는 $LOG_DIR 에 있습니다."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------------
# 3. 데이터 수집 로거 실행 (핵심: stdbuf -oL 필수 적용)
# ---------------------------------------------------------------------------------
echo ">> [Logger] 데이터 수집기 백그라운드 실행 시작..."

# [3-1] RTT (Latency)
# stdbuf -oL: 라인 단위 버퍼링 (즉시 기록)
stdbuf -oL /usr/share/bcc/tools/tcplife -T > "$LOG_DIR/rtt_log.txt" &
PIDS+=($!)

# [3-2] TCP Retransmission
stdbuf -oL bpftrace -e '
tracepoint:tcp:tcp_retransmit_skb { @[comm] = count(); } 
interval:s:1 { time("%H:%M:%S "); print(@); clear(@); }
' > "$LOG_DIR/retrans_bpftrace.log" &
PIDS+=($!)

# [3-3] Packet Count
stdbuf -oL bpftrace -e '
tracepoint:net:net_dev_xmit { @[comm] = count(); } 
interval:s:1 { time("%H:%M:%S "); print(@); clear(@); }
' > "$LOG_DIR/packets_count.log" &
PIDS+=($!)

# [3-4] Throughput
stdbuf -oL bpftrace -e '
tracepoint:net:net_dev_queue { @tx_bytes = sum(args->len); }
tracepoint:net:netif_receive_skb { @rx_bytes = sum(args->len); }
interval:s:1 { 
    time("%H:%M:%S "); 
    print(@tx_bytes); 
    print(@rx_bytes); 
    clear(@tx_bytes); clear(@rx_bytes); 
}
' > "$LOG_DIR/throughput.log" &
PIDS+=($!)

# [3-5] Memory Events
stdbuf -oL bpftrace -e '
tracepoint:kmem:kmalloc { @[comm] = count(); } 
interval:s:1 { time("%H:%M:%S "); print(@); clear(@); }
' > "$LOG_DIR/mem_event.log" &
PIDS+=($!)

# [3-6] Memory Usage (Polling Loop)
(
    echo "timestamp,total_MB,used_MB,free_MB,shared_MB,buff_cache_MB,available_MB"
    for i in $(seq 1 "$MAX_DURATION"); do
        ts="$(date +%F\ %T.%3N)"
        read _ total used free shared buff_cache available < <(free -m | awk '/^Mem:/ {print $1,$2,$3,$4,$5,$6,$7}')
        echo "$ts,$total,$used,$free,$shared,$buff_cache,$available"
        sleep 1
    done
) > "$LOG_DIR/mem_usage.csv" &
PIDS+=($!)

# [3-7] Conn Failures
(
    echo "# timestamp Tcp_line Netstat_line"
    for i in $(seq 1 "$MAX_DURATION"); do
        ts="$(date +%F\ %T.%3N)"
        tcp_line="$(grep '^Tcp:' /proc/net/snmp | tail -n 1)"
        tcpext_line="$(grep '^TcpExt:' /proc/net/netstat | tail -n 1 || true)"
        echo "$ts SNMP $tcp_line"
        echo "$ts NETSTAT $tcpext_line"
        sleep 1
    done
) > "$LOG_DIR/conn_fail_stats.log" &
PIDS+=($!)

# [3-8] TCP Socket Detail
(
    for i in $(seq 1 "$MAX_DURATION"); do
        echo "===== $(date +%F\ %T) ====="
        ss -intnp
        echo
        sleep 1
    done
) > "$LOG_DIR/tcp_ss_detail.log" &
PIDS+=($!)

# [3-9] CPU Run Queue (vmstat) 
# stdbuf -oL 사용 + awk 내부 fflush() 사용으로 실시간 기록 보장
(
    stdbuf -oL vmstat -n 1 "$MAX_DURATION" | awk '
    NR>2 { 
        # 1. 시스템 시간 가져오기 (외부 date 호출보다 빠름)
        ts = strftime("%H:%M:%S");
        
        # 2. 구분선 출력 (파서 인식용)
        print "===== " ts " =====";
        
        # 3. 데이터 출력
        OFS=",";
        print ts, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17;
        
        # 4. [중요] 버퍼 비우기 (파일에 즉시 쓰기)
        fflush();
    }' 
) > "$LOG_DIR/vmstat.log" &
PIDS+=($!)


# [3-10] SoftIRQ
(
    echo "# timestamp /proc/softirqs"
    for i in $(seq 1 "$MAX_DURATION"); do
        ts="$(date +%F\ %T.%3N)"
        echo "===== $ts ====="
        cat /proc/softirqs
        echo
        sleep 1
    done
) > "$LOG_DIR/softirqs.log" &
PIDS+=($!)

# [3-11] I/O Latency (iostat)
# iostat 버퍼링 문제 해결을 위해 stdbuf 사용
(
    echo "# timestamp iostat"
    stdbuf -oL iostat -x 1 "$MAX_DURATION"
) > "$LOG_DIR/iostat.log" &
PIDS+=($!)

# [3-12] CPU Throttling
if [ -n "$CPU_STAT_FILE" ]; then
    (
        echo "# timestamp cpu.stat"
        for i in $(seq 1 "$MAX_DURATION"); do
            ts="$(date +%F\ %T.%3N)"
            echo "===== $ts ====="
            cat "$CPU_STAT_FILE"
            echo
            sleep 1
        done
    ) > "$LOG_DIR/cpu_throttle.log" &
    PIDS+=($!)
fi

# ---------------------------------------------------------------------------------
# 4. 시나리오 실행
# ---------------------------------------------------------------------------------
echo ">> [Ready] 모든 수집기가 준비되었습니다. 5초 후 시나리오를 시작합니다..."
sleep 5

echo "=================================================================="
echo "🚀 [START] 시나리오 실행 시작"
echo "=================================================================="

# 👇 시나리오 스크립트 실행 👇
./scenario_0806.sh

echo "=================================================================="
echo "🏁 [END] 시나리오 실행 종료."
echo "=================================================================="

if [ -f "scenario_timeline.txt" ]; then
    mv "scenario_timeline.txt" "$LOG_DIR/"
    echo ">> 타임라인 파일을 이동했습니다."
fi

exit 0