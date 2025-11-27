import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

# ===============================================================
# 1. 설정
# ===============================================================
FINAL_COLS = [
    'time',
    'RTT_ms_mean', 'packet_loss_rate_pct', 'retrans_segs_per_sec', 'conn_fail_rate_server',
    'listen_drops_per_sec', 'Recv-Q_Bytes_max', 'Send-Q_Bytes_max', 'run_queue_len_cpu0',
    'cwnd_min', 'ssthresh_min',
    'nic_usage_pct', 'total_packets_per_sec', 'used_MB', 'available_MB',
    'total_mem_events_per_sec', 'cpu_throttle_per_sec',
    'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec', 'iostat_r_await', 'iostat_aqu_sz',
    'Label'
]
NIC_CAPACITY_BPS = 1_000_000_000  # 1Gbps

def clean_line(line):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line).strip()

def clip_negatives(df):
    if df.empty: return df
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0)
    return df

# ===============================================================
# 2. 파싱 로직 (기존 타임스탬프 기반 유지)
# ===============================================================
def parse_bpftrace_sum_by_time(filepath):
    """
    형식: 11:37:27 @[process]: 5
    같은 시간대의 모든 프로세스 카운트를 합산하여 반환
    """
    data_map = defaultdict(int)
    last_ts = None
    
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = clean_line(line)
                # 1. 시간 파싱
                ts_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                if ts_match:
                    last_ts = ts_match.group(1)
                
                # 2. 값 파싱 (라인 끝에 있는 숫자 추출)
                if last_ts and ":" in line:
                    val_match = re.search(r":\s*(\d+)", line)
                    if val_match:
                        data_map[last_ts] += int(val_match.group(1))
                        
        # 딕셔너리를 DataFrame으로 변환
        return pd.DataFrame(list(data_map.items()), columns=['time', 'val']).sort_values('time')
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()
    
def parse_bpftrace_logs(log_dir):
    # 1. NIC Throughput (기존 유지)
    tp_data = []
    try:
        with open(f"{log_dir}/throughput.log", "r") as f:
            lines = f.readlines()
        curr_ts, tx, rx = None, 0, 0
        for line in lines:
            line = clean_line(line)
            time_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
            if time_match:
                if curr_ts:
                    usage_pct = ((tx + rx) * 8 / NIC_CAPACITY_BPS) * 100
                    tp_data.append({'time': curr_ts, 'nic_usage_pct': usage_pct})
                curr_ts = time_match.group(1)
                tx, rx = 0, 0
            if "tx_bytes" in line:
                val = re.search(r":\s*(\d+)", line)
                if val: tx = int(val.group(1))
            elif "rx_bytes" in line:
                val = re.search(r":\s*(\d+)", line)
                if val: rx = int(val.group(1))
        if curr_ts:
             usage_pct = ((tx + rx) * 8 / NIC_CAPACITY_BPS) * 100
             tp_data.append({'time': curr_ts, 'nic_usage_pct': usage_pct})
    except: pass
    df_nic = pd.DataFrame(tp_data).groupby('time').mean().reset_index() if tp_data else pd.DataFrame()

    # 2. Memory Events (기존 유지)
    df_mem = parse_bpftrace_sum_by_time(f"{log_dir}/mem_event.log")
    if not df_mem.empty:
        df_mem = df_mem.rename(columns={'val': 'total_mem_events_per_sec'})

    # =========================================================
    # 3. [핵심 수정] Packet Loss Rate (eBPF 기반)
    # =========================================================
    # 전체 패킷 수 (TX + RX 등, packets_count.log)
    df_pkt_count = parse_bpftrace_sum_by_time(f"{log_dir}/packets_count.log")
    
    # 재전송 패킷 수 (retrans_bpftrace.log)
    df_retrans = parse_bpftrace_sum_by_time(f"{log_dir}/retrans_bpftrace.log")

    # 두 데이터프레임 병합 (Packet Loss 계산을 위해)
    df_loss = pd.DataFrame()
    if not df_pkt_count.empty:
        # 컬럼명 변경
        df_pkt_count = df_pkt_count.rename(columns={'val': 'total_packets_per_sec'})
        
        if not df_retrans.empty:
            df_retrans = df_retrans.rename(columns={'val': 'retrans_count'})
            # 시간 기준으로 병합 (Outer join 후 NaN은 0 처리)
            merged = pd.merge(df_pkt_count, df_retrans, on='time', how='left').fillna(0)
            
            # Loss Rate 계산: (재전송 / 전체패킷) * 100
            # 분모가 0이면 0 처리
            merged['packet_loss_rate_pct'] = merged.apply(
                lambda row: (row['retrans_count'] / row['total_packets_per_sec'] * 100) 
                if row['total_packets_per_sec'] > 0 else 0, axis=1
            )
            merged['retrans_segs_per_sec'] = merged['retrans_count'] # 초당 재전송 수
            df_loss = merged[['time', 'total_packets_per_sec', 'packet_loss_rate_pct', 'retrans_segs_per_sec']]
        else:
            # 재전송 로그가 없으면 Loss는 0으로 가정
            df_pkt_count['packet_loss_rate_pct'] = 0
            df_pkt_count['retrans_segs_per_sec'] = 0
            df_loss = df_pkt_count

    return df_nic, df_loss, df_mem

def parse_ss_log(log_dir):
    results = []
    try:
        with open(f"{log_dir}/tcp_ss_detail.log", "r") as f:
            lines = f.readlines()
        current_time = None
        recv_q, send_q, cwnd, ssthresh = [], [], [], []
        for line in lines:
            line = clean_line(line)
            if line.startswith("====="):
                if current_time:
                    results.append({
                        'time': current_time,
                        'Recv-Q_Bytes_max': max(recv_q) if recv_q else 0,
                        'Send-Q_Bytes_max': max(send_q) if send_q else 0,
                        'cwnd_min': min(cwnd) if cwnd else 0,
                        'ssthresh_min': min(ssthresh) if ssthresh else 0
                    })
                time_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                if time_match: current_time = time_match.group(1)
                recv_q, send_q, cwnd, ssthresh = [], [], [], []
            elif "ESTAB" in line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        recv_q.append(int(parts[1]))
                        send_q.append(int(parts[2]))
                    except: pass
            if "cwnd:" in line:
                c = re.search(r'cwnd:(\d+)', line)
                s = re.search(r'ssthresh:(\d+)', line)
                if c: cwnd.append(int(c.group(1)))
                if s: ssthresh.append(int(s.group(1)))
        if current_time:
             results.append({
                'time': current_time,
                'Recv-Q_Bytes_max': max(recv_q) if recv_q else 0,
                'Send-Q_Bytes_max': max(send_q) if send_q else 0,
                'cwnd_min': min(cwnd) if cwnd else 0,
                'ssthresh_min': min(ssthresh) if ssthresh else 0
            })
    except: pass
    return pd.DataFrame(results)

def parse_simple_logs(log_dir):
    # Mem Usage (타임스탬프 신뢰)
    try:
        df_mem = pd.read_csv(f"{log_dir}/mem_usage.csv")
        df_mem['time'] = df_mem['timestamp'].astype(str).apply(lambda x: re.search(r"(\d{2}:\d{2}:\d{2})", x).group(1) if re.search(r"\d{2}:\d{2}:\d{2}", x) else None)
        df_mem = df_mem.dropna(subset=['time'])
        df_mem = df_mem[['time', 'used_MB', 'available_MB']]
    except: df_mem = pd.DataFrame()
    
    # RTT (타임스탬프 신뢰)
    rtt_sums = defaultdict(float)
    rtt_counts = defaultdict(int)
    try:
        with open(f"{log_dir}/rtt_log.txt", "r") as f:
            next(f); next(f)
            for line in f:
                parts = line.split()
                if len(parts) > 5:
                    time_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                    if time_match:
                        t = time_match.group(1)
                        try:
                            ms = float(parts[-1])
                            rtt_sums[t] += ms
                            rtt_counts[t] += 1
                        except: pass
    except: pass
    rtt_final = [{'time': t, 'RTT_ms_mean': s/rtt_counts[t]} for t, s in rtt_sums.items()]
    df_rtt = pd.DataFrame(rtt_final) if rtt_final else pd.DataFrame()
    
    return df_mem, df_rtt
def parse_proc_counters(log_dir):
    # 1. SNMP & Netstat (ListenDrops 파싱)
    snmp_data = []
    try:
        with open(f"{log_dir}/conn_fail_stats.log", "r") as f:
            for line in f:
                parts = line.split()
                # 시간 파싱
                if len(parts) < 5: continue
                ts = parts[1].split(".")[0] 

                # A. SNMP 라인
                if "SNMP" in line and "Tcp:" in line:
                    try:
                        vals = [int(x) for x in parts[4:]]
                        if len(vals) > 7:
                            snmp_data.append({
                                'time': ts,
                                'type': 'snmp',
                                'conn_fail_rate_server': vals[6] 
                            })
                    except: pass
                
                # B. NETSTAT 라인
                elif "NETSTAT" in line and "TcpExt:" in line:
                    try:
                        vals = [int(x) for x in parts[4:]]
                        if len(vals) > 20:
                            snmp_data.append({
                                'time': ts,
                                'type': 'netstat',
                                'listen_drops': vals[20]
                            })
                    except: pass

    except Exception as e:
        print(f"Error parsing conn_fail_stats: {e}")

    # 데이터 정리
    df_raw = pd.DataFrame(snmp_data)
    
    # [수정 핵심] 초기화 및 컬럼 명시적 분리 (중복 방지)
    df_snmp = pd.DataFrame()
    df_netstat = pd.DataFrame()

    if not df_raw.empty:
        # SNMP 데이터 분리 (컬럼이 있을 때만)
        if 'conn_fail_rate_server' in df_raw.columns:
            temp_snmp = df_raw[df_raw['type'] == 'snmp']
            if not temp_snmp.empty:
                df_snmp = temp_snmp[['time', 'conn_fail_rate_server']].drop_duplicates('time', keep='last').set_index('time')

        # Netstat 데이터 분리 (컬럼이 있을 때만)
        if 'listen_drops' in df_raw.columns:
            temp_net = df_raw[df_raw['type'] == 'netstat']
            if not temp_net.empty:
                df_netstat = temp_net[['time', 'listen_drops']].drop_duplicates('time', keep='last').set_index('time')
    
    # 병합 (Outer Join)
    df_final = pd.concat([df_snmp, df_netstat], axis=1).fillna(0).sort_index().reset_index()

    # Diff 계산 (누적값 -> 초당 변화량)
    cols_to_diff = []
    if 'conn_fail_rate_server' in df_final.columns: cols_to_diff.append('conn_fail_rate_server')
    if 'listen_drops' in df_final.columns: cols_to_diff.append('listen_drops')

    for col in cols_to_diff:
        # 여기서 이제 중복 컬럼 오류가 발생하지 않습니다.
        df_final[f'{col}_per_sec'] = df_final[col].diff().fillna(0).clip(lower=0)

    # 최종 컬럼 이름 변경
    df_final = df_final.rename(columns={'listen_drops_per_sec': 'listen_drops_per_sec'})
    
    # 필요한 컬럼만 선택하여 반환
    final_cols = ['time']
    if 'listen_drops_per_sec' in df_final.columns: final_cols.append('listen_drops_per_sec')
    if 'conn_fail_rate_server_per_sec' in df_final.columns: final_cols.append('conn_fail_rate_server_per_sec')
    
    df_result = df_final[final_cols] if not df_final.empty else pd.DataFrame()

    # 2. SoftIRQ (기존 로직 유지)
    sirq_data = []
    try:
        if os.path.exists(f"{log_dir}/softirqs.log"):
            with open(f"{log_dir}/softirqs.log", "r") as f:
                content = f.read()
            blocks = re.split(r"={5,}\s+\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})\.\d+\s+={5,}", content)
            for i in range(1, len(blocks), 2):
                ts = blocks[i]
                data_block = blocks[i+1]
                rx_sum, tx_sum = 0, 0
                for line in data_block.strip().split('\n'):
                    if "NET_RX:" in line: rx_sum = sum(int(x) for x in re.findall(r'\d+', line))
                    elif "NET_TX:" in line: tx_sum = sum(int(x) for x in re.findall(r'\d+', line))
                if rx_sum > 0 or tx_sum > 0:
                    sirq_data.append({'time': ts, 'rx': rx_sum, 'tx': tx_sum})
    except: pass
    
    df_sirq = pd.DataFrame(sirq_data) 
    if not df_sirq.empty:
        df_sirq['softirq_net_rx_per_sec'] = df_sirq['rx'].diff().fillna(0).clip(lower=0)
        df_sirq['softirq_net_tx_per_sec'] = df_sirq['tx'].diff().fillna(0).clip(lower=0)
        df_sirq = df_sirq[['time', 'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec']]

    return df_result, df_sirq
    
def parse_throttle(log_dir):
    th_data = []
    try:
        with open(f"{log_dir}/cpu_throttle.log", "r") as f:
            lines = f.readlines()
        curr_ts = None
        for line in lines:
            line = clean_line(line)
            if line.startswith("====="):
                time_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                if time_match: curr_ts = time_match.group(1)
            elif "nr_throttled" in line and curr_ts:
                val = int(line.split()[1])
                th_data.append({'time': curr_ts, 'val': val})
    except: pass
    
    df_th = pd.DataFrame(th_data)
    if not df_th.empty:
        df_th['cpu_throttle_per_sec'] = df_th['val'].diff().fillna(0)
        df_th = df_th[['time', 'cpu_throttle_per_sec']]
        df_th = clip_negatives(df_th)
    return df_th

# ===============================================================
# 3. [핵심] Vmstat & Iostat 
# ===============================================================

def parse_vmstat_ignore_time(log_dir):
    """
    vmstat.log는 타임스탬프가 꼬였으므로, 순서대로 읽어서 리스트로 반환
    """
    run_queue_list = []
    try:
        with open(f"{log_dir}/vmstat.log", "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = clean_line(line).strip()
            # 구분선 무시
            if not line or "=====" in line or "timestamp" in line or "swpd" in line:
                continue
            
            parts = [x for x in line.split(',') if x.strip() != '']
            # 데이터 유효성 검사 (최소 2개 이상: 시간, r, ...)
            if len(parts) >= 2:
                try:
                    # 첫 번째는 꼬인 시간이므로 무시, 두 번째(r) 값 가져옴
                    r_val = int(parts[1])
                    run_queue_list.append(r_val)
                except: continue
    except Exception as e: print(f"Vmstat Error: {e}")
    
    return run_queue_list

def parse_iostat_ignore_time(log_dir):

    r_await_list = []
    aqu_sz_list = []
    try:
        with open(f"{log_dir}/iostat.log", "r") as f:
            for line in f:
                # 메인 디스크(sda 또는 nvme)만 필터링
                if "sda" in line or "nvme" in line:
                    parts = line.split()
                    if len(parts) >= 14:
                        try:
                            r_await_list.append(float(parts[-6]))
                            aqu_sz_list.append(float(parts[-4]))
                        except: pass
    except: pass
    
    return r_await_list, aqu_sz_list

def apply_timeline_labels(df, log_dir):
    timeline_path = f"{log_dir}/scenario_timeline.txt"
    df['Label'] = 0 
    if not os.path.exists(timeline_path): return df
    
    phases = []
    try:
        with open(timeline_path, "r") as f:
            for line in f:
                if "," in line:
                    ts, lbl = line.strip().split(",")
                    if re.search(r"\d{2}:\d{2}:\d{2}", ts):
                        phases.append({'time': ts.strip(), 'label': 1 if 'ATTACK' in lbl.upper() else 0})
    except: pass
    phases.sort(key=lambda x: x['time'])
    
    for i in range(len(phases)):
        start_t = phases[i]['time']
        label_val = phases[i]['label']
        end_t = phases[i+1]['time'] if i < len(phases) - 1 else "23:59:59"
        df.loc[(df['time'] >= start_t) & (df['time'] < end_t), 'Label'] = label_val
        
    return df

# ===============================================================
# 4. 통합 실행 (Hybrid Merge)
# ===============================================================

def process_log_directory(log_dir):
    print(f"\nProcessing: {log_dir} (Hybrid Mode)")
    
    # [Step 1] Time 기준 병합할 데이터들 로드
    df_nic, df_pkt, df_mem_evt = parse_bpftrace_logs(log_dir)
    df_ss = parse_ss_log(log_dir)
    df_snmp, df_sirq = parse_proc_counters(log_dir)
    df_mem_usage, df_rtt = parse_simple_logs(log_dir)
    df_th = parse_throttle(log_dir)
    
    time_based_dfs = [df_nic, df_pkt, df_mem_evt, df_ss, df_snmp, df_sirq, df_mem_usage, df_rtt, df_th]
    names = ['NIC', 'PKT', 'MEM_EVT', 'SS', 'SNMP', 'SIRQ', 'MEM_USE', 'RTT', 'THROT']
    
    # 유효한 데이터프레임 필터링
    valid_dfs = [d for d in time_based_dfs if not d.empty and 'time' in d.columns]
    
    if not valid_dfs:
        print("❌ Error: 시간 기반 데이터가 하나도 없습니다.")
        return

    # [Step 2] Time 기준 Outer Join (기준 뼈대 만들기)
    final_df = valid_dfs[0]
    for df in valid_dfs[1:]:
        final_df = pd.merge(final_df, df, on='time', how='outer')
    
    # 시간순 정렬 및 결측치 0 처리
    final_df = final_df.sort_values('time').fillna(0).reset_index(drop=True)
    print(f"✅ Base Time-Merged DataFrame created: {len(final_df)} rows")

    # [Step 3] Vmstat & Iostat 순서대로 붙이기 (Zipper Merge)
    vmstat_vals = parse_vmstat_ignore_time(log_dir)
    io_await, io_aqu = parse_iostat_ignore_time(log_dir)
    
    # 길이 맞추기 (final_df 길이만큼만 앞에서부터 자름)
    target_len = len(final_df)
    
    # vmstat 붙이기
    vmstat_col = vmstat_vals[:target_len]
    # 만약 vmstat 데이터가 부족하면 0으로 채움
    if len(vmstat_col) < target_len:
        vmstat_col += [0] * (target_len - len(vmstat_col))
    final_df['run_queue_len_cpu0'] = vmstat_col
    
    # iostat 붙이기
    await_col = io_await[:target_len]
    aqu_col = io_aqu[:target_len]
    if len(await_col) < target_len: await_col += [0] * (target_len - len(await_col))
    if len(aqu_col) < target_len: aqu_col += [0] * (target_len - len(aqu_col))
        
    final_df['iostat_r_await'] = await_col
    final_df['iostat_aqu_sz'] = aqu_col
    
    print(f"✅ Vmstat({len(vmstat_vals)}) & Iostat({len(io_await)}) merged by index")

    # [Step 4] 라벨링 및 저장
    final_df = apply_timeline_labels(final_df, log_dir)
    
    for col in FINAL_COLS:
        if col not in final_df.columns: final_df[col] = 0
    final_df = final_df[FINAL_COLS]
    
    output_name = f"{log_dir}_final_hybrid_new.csv"
    final_df.to_csv(output_name, index=False)
    print(f"✅ Final Saved: {output_name}")

if __name__ == "__main__":
    TARGET_DIR = "./0806/AI_TRAINING_DATA_20251122_123712" 
    process_log_directory(TARGET_DIR)