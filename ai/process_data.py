import pandas as pd
import numpy as np
import glob
import joblib
import os
from sklearn.preprocessing import MinMaxScaler


def process_data_by_file(input_pattern="data/*.csv", timesteps=60, predict_shift=10):
    FEATURE_COLS = [
        'RTT_ms_mean', 'packet_loss_rate_pct', 'retrans_segs_per_sec',
        'conn_fail_rate_server', 'listen_drops_per_sec',
        'Recv-Q_Bytes_max', 'Send-Q_Bytes_max', 'run_queue_len_cpu0',
        'cwnd_min', 'ssthresh_min', 'nic_usage_pct', 'total_packets_per_sec',
        'used_MB', 'available_MB', 'total_mem_events_per_sec',
        'cpu_throttle_per_sec', 'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec',
        'iostat_r_await', 'iostat_aqu_sz'
    ]
    TARGET_COL = "RTT_ms_mean"

    print(f"\n{'=' * 40}")
    print(f"⚙️ [데이터 전처리] Future Max Prediction 모드")
    print(f"{'=' * 40}")

    # 1. 모든 파일 로드 및 유효성 검사
    all_files = sorted(glob.glob(input_pattern))
    valid_dfs = []

    print(f" - 타겟 설정: 미래 {predict_shift}초 구간 내 최대값(Max) 예측")
    print(" - 파일 로드 중...")

    for f in all_files:
        try:
            temp = pd.read_csv(f)
            # 컬럼 존재 여부 확인
            valid_cols = [c for c in FEATURE_COLS if c in temp.columns]
            if TARGET_COL not in valid_cols:
                continue

            df_subset = temp[valid_cols].copy()

            # 설명:
            # 1. rolling(window=10).max(): 현재 행 포함 '과거' 10개의 최대값을 구함
            # 2. shift(-10): 이 계산된 값을 10칸 뒤로 당김 (즉, 미래의 결과를 현재로 가져옴)
            # 결과적으로 현재 시점 t의 y_target은 t+1 ~ t+10 구간의 최대값이 됨

            future_max = df_subset[TARGET_COL].rolling(window=predict_shift).max().shift(-predict_shift)
            df_subset['y_target'] = future_max

            df_subset = df_subset.dropna()  # NaN 제거

            # 데이터가 윈도우 크기보다 작으면 스킵 (시퀀스를 못 만듦)
            if len(df_subset) <= timesteps:
                print(f"   [Skip] 데이터 부족 ({len(df_subset)} rows): {f}")
                continue

            valid_dfs.append(df_subset)

        except Exception as e:
            print(f"   [Error] {f}: {e}")

    if not valid_dfs:
        print("[오류] 유효한 데이터가 없습니다. (경로 확인 필요: augmented_data/*.csv)")
        return

    # 2. 스케일러 학습 (전체 데이터 기준)
    full_concat_for_scaler = pd.concat(valid_dfs, ignore_index=True)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 전체 데이터로 fitting
    scaler_X.fit(full_concat_for_scaler[FEATURE_COLS].values)
    scaler_y.fit(full_concat_for_scaler[['y_target']].values)

    del full_concat_for_scaler
    print(" - 스케일러 학습 완료 (전체 데이터 기준)")

    # 3. 파일별 시퀀스 생성 함수
    def create_sequences(X_data, y_data, time_steps):
        Xs, ys = [], []
        for i in range(len(X_data) - time_steps):
            Xs.append(X_data[i: i + time_steps])
            # 타겟: 시퀀스 끝나는 시점의 y_target 값 (이미 미래 max로 shift 되어있음)
            ys.append(y_data[i + time_steps - 1])
        return np.array(Xs), np.array(ys)

    X_list = []
    y_list = []

    print(f" - {len(valid_dfs)}개 파일에 대해 시퀀스 변환 중...")

    for df in valid_dfs:
        # Transform
        scaled_x = scaler_X.transform(df[FEATURE_COLS].values)
        scaled_y = scaler_y.transform(df[['y_target']].values)

        # 시퀀스 생성
        seq_x, seq_y = create_sequences(scaled_x, scaled_y, timesteps)

        if len(seq_x) > 0:
            X_list.append(seq_x)
            y_list.append(seq_y)

    # 4. 전체 시퀀스 병합
    if not X_list:
        print("[오류] 시퀀스 생성 실패 (데이터 부족)")
        return

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    print(f" - 전체 시퀀스 병합 완료: {X_all.shape}")

    # 5. Train/Test Split (시간 순서 유지)
    split_idx = int(len(X_all) * 0.8)

    X_train_lstm = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_test_lstm = X_all[split_idx:]
    y_test = y_all[split_idx:]

    # 6. XGBoost용 Flatten
    def flatten_for_xgb(X_3d):
        nsamples, nsteps, nfeatures = X_3d.shape
        return X_3d.reshape((nsamples, nsteps * nfeatures))

    X_train_xgb = flatten_for_xgb(X_train_lstm)
    X_test_xgb = flatten_for_xgb(X_test_lstm)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # 7. 저장
    joblib.dump(scaler_X, 'scaler_X.joblib')
    joblib.dump(scaler_y, 'scaler_y.joblib')

    np.save('X_train_lstm.npy', X_train_lstm)
    np.save('X_test_lstm.npy', X_test_lstm)
    np.save('X_train_xgb.npy', X_train_xgb)
    np.save('X_test_xgb.npy', X_test_xgb)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # print(f"✅ 처리 완료!")
    # print(f"   LSTM Train: {X_train_lstm.shape}, Test: {X_test_lstm.shape}")


if __name__ == "__main__":
    # 실행 시 input 패턴이 augmented_data를 바라보도록 수정
    process_data_by_file(input_pattern="data/*.csv")