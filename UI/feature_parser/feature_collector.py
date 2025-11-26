import subprocess
import threading
import re
import time
import os
from datetime import datetime
from collections import defaultdict

# =========================================================
# [설정] 환경에 맞게 수정
# =========================================================
TARGET_NIC = "ens33"
TARGET_DISK = "sda"

# 출력용 단위 매핑
UNIT_MAP = {
    'cpu_nr_throttled': 'count',
    'disk_io_weighted_ms': 'ms',
    'disk_read_ios': 'count',
    'disk_read_time_ms': 'ms',
    'ebpf_mem_events': 'events',
    'ebpf_packet_count': 'pkts',
    'ebpf_retrans_count': 'count',
    'ebpf_tx_attempts': 'count',
    'mem_avail_mb': 'MB',
    'mem_total_mb': 'MB',
    'nic_rx_bytes': 'bytes',
    'nic_tx_bytes': 'bytes',
    'sched_run_queue': 'count',
    'snmp_ActiveOpens': 'count',
    'snmp_AttemptFails': 'count',
    'snmp_ListenDrops': 'count',
    'snmp_PassiveOpens': 'count',
    'snmp_RetransSegs': 'count',
    'softirq_net_rx': 'count',
    'softirq_net_tx': 'count',
    'tcp_cwnd_min': 'segments',
    'tcp_recv_q_max': 'bytes',
    'tcp_rtt_count': 'count',
    'tcp_rtt_sum': 'ms',
    'tcp_send_q_max': 'bytes',
    'tcp_ssthresh_min': 'segments',
    'timestamp': ''
}


class FinalRawCollector:
    def __init__(self):
        self.ebpf_storage = defaultdict(int)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self._start_unified_bpftrace()

    def _start_unified_bpftrace(self):
        # Kprobe 사용
        bpftrace_script = """
        kprobe:tcp_retransmit_skb { @retrans_count = count(); }
        tracepoint:net:net_dev_queue { @tx_attempts = count(); }
        tracepoint:net:net_dev_xmit { @packet_count = count(); }
        tracepoint:kmem:kmalloc { @mem_events = count(); }
        interval:s:1 { 
            print(@retrans_count); print(@tx_attempts); 
            print(@packet_count); print(@mem_events); 
        }
        """

        def _worker():
            cmd = ["bpftrace", "-e", bpftrace_script]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
            )
            pattern = re.compile(r"@(\w+): (\d+)")
            for line in process.stdout:
                if self.stop_event.is_set(): break
                match = pattern.search(line)
                if match:
                    with self.lock:
                        self.ebpf_storage[match.group(1)] = int(match.group(2))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _read_file(self, path):
        try:
            with open(path, 'r') as f:
                return f.read()
        except:
            return ""

    def _run_cmd(self, cmd):
        try:
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=0.5).stdout
        except:
            return ""

    def get_snapshot(self):
        data = {}
        data['timestamp'] = datetime.now().isoformat()

        # -----------------------------------------------------
        # [1] SS 명령어: Min/Max/Mean 계산을 위한 Raw Data
        # -----------------------------------------------------
        ss_out = self._run_cmd(["ss", "-ti"])

        # 리스트 초기화
        rtt_list = []
        cwnd_list = []
        ssthresh_list = []
        recv_q_list = []
        send_q_list = []

        if ss_out:
            lines = ss_out.splitlines()[1:]  # 헤더 제외
            full_text = " ".join(lines)

            # RTT 수집 (Mean 계산용 Sum/Count)
            rtt_matches = re.findall(r"\s+rtt:([\d.]+)(?:/[\d.]+)?", full_text)
            #rtt_matches = re.findall(r"rtt:([\d.]+)", full_text)
            rtt_list = [float(x) for x in rtt_matches]

            # Cwnd & Ssthresh 수집 (Min 계산용)
            cwnd_matches = re.findall(r"cwnd:(\d+)", full_text)
            cwnd_list = [int(x) for x in cwnd_matches]

            ssthresh_matches = re.findall(r"ssthresh:(\d+)", full_text)
            ssthresh_list = [int(x) for x in ssthresh_matches]

            # Recv-Q & Send-Q 수집 (Max 계산용)
            for line in lines:
                parts = line.split()
                # State가 ESTAB, LISTEN, TIME-WAIT 등인 경우만
                if len(parts) >= 5 and parts[0] in ['ESTAB', 'LISTEN', 'TIME-WAIT', 'CLOSE-WAIT']:
                    if parts[1].isdigit(): recv_q_list.append(int(parts[1]))
                    if parts[2].isdigit(): send_q_list.append(int(parts[2]))

        # Raw Data 추출
        # RTT Mean용
        data['tcp_rtt_sum'] = sum(rtt_list)
        data['tcp_rtt_count'] = len(rtt_list)

        # Max/Min 값 (리스트가 비어있으면 0 처리)
        data['tcp_recv_q_max'] = max(recv_q_list) if recv_q_list else 0
        data['tcp_send_q_max'] = max(send_q_list) if send_q_list else 0
        data['tcp_cwnd_min'] = min(cwnd_list) if cwnd_list else 0
        data['tcp_ssthresh_min'] = min(ssthresh_list) if ssthresh_list else 0

        # -----------------------------------------------------
        # [2] SNMP & Netstat (누적값 - Diff 필요)
        # -----------------------------------------------------
        snmp = self._read_file("/proc/net/snmp")
        netstat = self._read_file("/proc/net/netstat")

        # 기본값
        data['snmp_ActiveOpens'] = 0
        data['snmp_PassiveOpens'] = 0
        data['snmp_AttemptFails'] = 0
        data['snmp_RetransSegs'] = 0
        data['snmp_ListenDrops'] = 0

        for line in snmp.splitlines():
            if line.startswith("Tcp:"):
                parts = line.split()
                if len(parts) > 12 and parts[1].isdigit():
                    data['snmp_ActiveOpens'] = int(parts[5])
                    data['snmp_PassiveOpens'] = int(parts[6])
                    data['snmp_AttemptFails'] = int(parts[7])
                    data['snmp_RetransSegs'] = int(parts[12])

        for line in netstat.splitlines():
            if line.startswith("TcpExt:"):
                parts = line.split()
                if len(parts) > 21 and parts[1].isdigit():
                    data['snmp_ListenDrops'] = int(parts[21])

        # -----------------------------------------------------
        # [3] NIC & Memory
        # -----------------------------------------------------
        net_dev = self._read_file("/proc/net/dev")
        data['nic_rx_bytes'] = 0
        data['nic_tx_bytes'] = 0
        for line in net_dev.splitlines():
            if TARGET_NIC in line:
                parts = line.split(":")[1].split()
                data['nic_rx_bytes'] = int(parts[0])
                data['nic_tx_bytes'] = int(parts[8])

        meminfo = self._read_file("/proc/meminfo")
        mem_total = 0
        mem_avail = 0
        for line in meminfo.splitlines():
            if "MemTotal" in line: mem_total = int(line.split()[1])
            if "MemAvailable" in line: mem_avail = int(line.split()[1])

        # MB 단위로 변환해서 제공 (Snapshot)
        data['mem_total_mb'] = mem_total / 1024
        data['mem_avail_mb'] = mem_avail / 1024

        # -----------------------------------------------------
        # [4] Disk Stats (iostat r_await, aqu_sz 계산용)
        # -----------------------------------------------------
        diskstats = self._read_file("/proc/diskstats")
        data['disk_read_ios'] = 0  # Field 4 (읽기 완료 횟수)
        data['disk_read_time_ms'] = 0  # Field 7 (읽기 소요 시간)
        data['disk_io_weighted_ms'] = 0  # Field 14 (가중치 시간 - Queue Size용)

        for line in diskstats.splitlines():
            parts = line.split()
            # parts[2]가 디스크 이름
            if len(parts) > 13 and parts[2] == TARGET_DISK:
                data['disk_read_ios'] = int(parts[3])
                data['disk_read_time_ms'] = int(parts[6])
                data['disk_io_weighted_ms'] = int(parts[13])

        # -----------------------------------------------------
        # [5] Scheduler & CPU & SoftIRQ (누적값)
        # -----------------------------------------------------
        data['sched_run_queue'] = os.getloadavg()[0]

        cpu_stat = self._read_file("/sys/fs/cgroup/cpu/cpu.stat") or self._read_file("/sys/fs/cgroup/cpu.stat")
        data['cpu_nr_throttled'] = 0
        if cpu_stat:
            for line in cpu_stat.splitlines():
                if "nr_throttled" in line: data['cpu_nr_throttled'] = int(line.split()[1])

        softirq = self._read_file("/proc/softirqs")
        data['softirq_net_rx'] = 0
        data['softirq_net_tx'] = 0
        for line in softirq.splitlines():
            if "NET_RX" in line:
                data['softirq_net_rx'] = sum(int(x) for x in line.split()[1:] if x.isdigit())
            elif "NET_TX" in line:
                data['softirq_net_tx'] = sum(int(x) for x in line.split()[1:] if x.isdigit())

        # -----------------------------------------------------
        # [6] eBPF 병합
        # -----------------------------------------------------
        with self.lock:
            data['ebpf_packet_count'] = self.ebpf_storage['packet_count']
            data['ebpf_retrans_count'] = self.ebpf_storage['retrans_count']
            data['ebpf_mem_events'] = self.ebpf_storage['mem_events']
            data['ebpf_tx_attempts'] = self.ebpf_storage['tx_attempts']

        return data

import os
from datetime import datetime

class FeatureTransformer:
    def __init__(self, nic_name="ens33"):
        self.prev = None
        self.prev_time = None
        self.link_speed = self._detect_nic_speed(nic_name)

    def _detect_nic_speed(self, nic):
        path = f"/sys/class/net/{nic}/speed"
        try:
            with open(path, "r") as f:
                mbps = int(f.read().strip())
                if mbps > 0:
                    print(f"[INFO] NIC {nic} Speed Detected: {mbps} Mbps")
                    return mbps * 1_000_000   # Mbps → bps 변환
        except:
            pass

        # 실패하면 기본값 1Gbps
        print(f"[WARN] NIC 속도 감지 실패 → 1Gbps 기본값 사용")
        return 1_000_000_000

    def transform(self, cur):
        now = datetime.fromisoformat(cur['timestamp'])

        if self.prev is None:
            self.prev = cur
            self.prev_time = now
            return {
                'RTT_ms_mean': cur['tcp_rtt_sum'] / cur['tcp_rtt_count'] if cur['tcp_rtt_count'] else 0,
                'Recv_Q_Bytes_max': cur['tcp_recv_q_max'],
                'Send_Q_Bytes_max': cur['tcp_send_q_max'],
                'cwnd_min': cur['tcp_cwnd_min'],
                'ssthresh_min': cur['tcp_ssthresh_min'],

                'packet_loss_rate_pct': 0,
                'retrans_segs_per_sec': 0,

                'conn_fail_rate_server': 0,
                'listen_drops_per_sec': 0,

                'iostat_r_await': 0,
                'iostat_aqu_sz': 0,

                'nic_usage_pct': 0,
                'total_packets_per_sec': 0,
                'total_mem_events_per_sec': 0,
                'cpu_throttle_per_sec': 0,
                'softirq_net_rx_per_sec': 0,
                'softirq_net_tx_per_sec': 0,

                'used_MB': cur['mem_total_mb'] - cur['mem_avail_mb'],
                'available_MB': cur['mem_avail_mb'],
                'run_queue_len_cpu0': cur['sched_run_queue']
            }

        prev = self.prev
        dt = (now - self.prev_time).total_seconds()
        dt_ms = dt * 1000
        if dt <= 0:
            dt = 1

        Δ = lambda key: cur[key] - prev[key]

        out = {}

        # == SNAPSHOT ==
        out['RTT_ms_mean'] = cur['tcp_rtt_sum'] / cur['tcp_rtt_count'] if cur['tcp_rtt_count'] else 0
        out['Recv_Q_Bytes_max'] = cur['tcp_recv_q_max']
        out['Send_Q_Bytes_max'] = cur['tcp_send_q_max']
        out['cwnd_min'] = cur['tcp_cwnd_min']
        out['ssthresh_min'] = cur['tcp_ssthresh_min']

        # == DIFF 기반 계산 ==
        tx = Δ('ebpf_tx_attempts')
        rt = Δ('ebpf_retrans_count')

        out['packet_loss_rate_pct'] = (rt / tx * 100) if tx > 0 else 0
        out['retrans_segs_per_sec'] = rt / dt

        ld = Δ('snmp_ListenDrops')
        po = Δ('snmp_PassiveOpens')
        denom = ld + po
        out['conn_fail_rate_server'] = (ld / denom * 100) if denom > 0 else 0
        out['listen_drops_per_sec'] = ld / dt

        read_ios = Δ('disk_read_ios')
        read_time = Δ('disk_read_time_ms')
        weighted = Δ('disk_io_weighted_ms')

        out['iostat_r_await'] = (read_time / read_ios) if read_ios > 0 else 0
        out['iostat_aqu_sz'] = weighted / dt_ms

        total_bytes = Δ('nic_rx_bytes') + Δ('nic_tx_bytes')
        out['nic_usage_pct'] = (total_bytes * 8 / (self.link_speed * dt) * 100)

        out['total_packets_per_sec'] = Δ('ebpf_packet_count') / dt
        out['total_mem_events_per_sec'] = Δ('ebpf_mem_events') / dt
        out['cpu_throttle_per_sec'] = Δ('cpu_nr_throttled') / dt
        out['softirq_net_rx_per_sec'] = Δ('softirq_net_rx') / dt
        out['softirq_net_tx_per_sec'] = Δ('softirq_net_tx') / dt

        # == Memory & RunQueue ==
        out['used_MB'] = cur['mem_total_mb'] - cur['mem_avail_mb']
        out['available_MB'] = cur['mem_avail_mb']
        out['run_queue_len_cpu0'] = cur['sched_run_queue']

        # 상태 업데이트
        self.prev = cur
        self.prev_time = now

        return out

# 실행부
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Error: root 권한 필요.")
        exit(1)

    collector = FinalRawCollector()

    # NIC 자동 감지 (ens33 기준)
    ft = FeatureTransformer(nic_name="ens33")

    while True:
        raw = collector.get_snapshot()
        feat = ft.transform(raw)
        for k, v in feat.items():
            print(f"{k}: {v:.5f}" if isinstance(v, (int, float)) else f"{k}: {v}")
        time.sleep(1)
