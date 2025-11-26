import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import traceback

# 1. 모델 로드
try:
    model = joblib.load('model_xgboost_best.joblib')
    print("✅ XGBoost 모델 로드 성공")
except:
    print("모델 파일이 없습니다.")
    print(traceback.format_exc())
    exit()

# 2. 피처 이름 (20개)
feature_names = [
    'RTT_ms_mean', 'packet_loss_rate_pct', 'retrans_segs_per_sec',
    'conn_fail_rate_server', 'listen_drops_per_sec',
    'Recv-Q_Bytes_max', 'Send-Q_Bytes_max', 'run_queue_len_cpu0',
    'cwnd_min', 'ssthresh_min', 'nic_usage_pct', 'total_packets_per_sec',
    'used_MB', 'available_MB', 'total_mem_events_per_sec',
    'cpu_throttle_per_sec', 'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec',
    'iostat_r_await', 'iostat_aqu_sz'
]

# 3. 중요도 추출 및 집계 (Aggregation)
# 모델은 1200개(60초*20개)의 중요도 값을 가짐
raw_importances = model.feature_importances_

if len(raw_importances) != 20:
    print(f"ℹ️ 모델 피처 개수({len(raw_importances)})가 20개가 아니므로, 20개로 집계합니다.")

    # 20개 공간 생성
    agg_importances = np.zeros(len(feature_names))

    # 1200개를 순회하며 해당하는 피처 인덱스에 더함
    # 구조: [T0_F0, T0_F1... T0_F19, T1_F0... ] 순서이므로 % 20으로 인덱스 찾음
    for i, score in enumerate(raw_importances):
        feat_idx = i % len(feature_names)
        agg_importances[feat_idx] += score

    importances = agg_importances
else:
    importances = raw_importances

# 정렬 (높은 순)
indices = np.argsort(importances)[::-1]

# 데이터프레임 만들기
df_imp = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices]
})

# 4. 시각화
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")  # 배경 깔끔하게

# 막대 그래프
ax = sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')

# 수치 표시
for i in ax.containers:
    ax.bar_label(i, fmt='%.4f', padding=3, fontsize=10)

plt.title('Feature Importance for RTT Prediction (Aggregated)', fontsize=15, fontweight='bold')
plt.xlabel('Total Importance Score (Sum over 60 steps)', fontsize=12)
plt.ylabel('eBPF Metrics', fontsize=12)
plt.tight_layout()

# 저장
plt.savefig('feature_importance_final.png', dpi=300)
print("✅ 중요도 그래프 저장 완료: feature_importance_final.png")
plt.show()

# 결과 출력
print("\n[Top 5 핵심 인자]")
print(df_imp.head(5))