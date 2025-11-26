import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

class rtt_predictor:
    _models = {}
    _scalers = {}

    # 피처 순서
    FEATURE_COLS = [
        'RTT_ms_mean', 'packet_loss_rate_pct', 'retrans_segs_per_sec',
        'conn_fail_rate_server', 'listen_drops_per_sec',
        'Recv_Q_Bytes_max', 'Send_Q_Bytes_max', 'run_queue_len_cpu0',
        'cwnd_min', 'ssthresh_min', 'nic_usage_pct', 'total_packets_per_sec',
        'used_MB', 'available_MB', 'total_mem_events_per_sec',
        'cpu_throttle_per_sec', 'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec',
        'iostat_r_await', 'iostat_aqu_sz'
    ]

    def __init__(self, data):
        self.sixty_sec_data = [data.copy() for _ in range(60)]

        # 인스턴스 생성 시 모델 로드 시도
        self._load_artifacts()

    @classmethod
    def _load_artifacts(cls):
        # 모델 로드
        if cls._models and cls._scalers:
            return True

        if not cls._scalers:
            try:
                cls._scalers['X'] = joblib.load('model/scaler_X.joblib')
                cls._scalers['y'] = joblib.load('model/scaler_y.joblib')
            except FileNotFoundError:
                print("[오류] 스케일러 파일(.joblib)이 없습니다.")
                return False

        if not cls._models:
            # XGBoost
            try:
                cls._models['xgboost'] = joblib.load('model/model_xgboost_best.joblib')
            except:
                print("[주의] XGBoost 모델 파일이 없습니다.")

            # LSTM
            try:
                cls._models['lstm'] = load_model('model/model_lstm_best.keras', compile=False)
            except:
                print("[주의] LSTM 모델 파일이 없습니다.")

            # Transformer
            try:
                cls._models['transformer'] = load_model('model/model_transformer_best.keras', compile=False)
            except:
                print("[주의] Transformer 모델 파일이 없습니다.")

        return True

    """
    @param list sixty_sec_data     csv랑 같은 형식의 데이터 60개
    @param str model     "lstm", "xgboost" 둘 중 하나 선택

    @return rtt값
    """

    def predict_data(self, data, model: str = "xgboost") -> float:
        if not self._models or not self._scalers:
            if not self._load_artifacts():
                return None

        self.sixty_sec_data.pop(0)
        self.sixty_sec_data.append(data)

        model = model.lower()
        if model not in self._models:
            print(f"[오류] '{model}' 모델이 로드되지 않았습니다.")
            return None

        df = pd.DataFrame(self.sixty_sec_data)

        # 학습할 때 쓴 컬럼만 골라내기 (순서 보장)
        try:
            df = df[self.FEATURE_COLS]
        except KeyError as e:
            print(f"[오류] 입력 데이터에 필수 컬럼이 누락되었습니다: {e}")
            return None

        input_scaled = self._scalers['X'].transform(df.values)

        if model == "xgboost":
            # 2D Flatten: (1, 1200)
            input_reshaped = input_scaled.reshape(1, -1)
            pred_scaled = self._models['xgboost'].predict(input_reshaped)

        elif model == "lstm":
            # 3D: (1, 60, 20)
            input_reshaped = input_scaled.reshape(1, 60, len(self.FEATURE_COLS))
            pred_scaled = self._models['lstm'].predict(input_reshaped, verbose=0)

        elif model == "transformer":
            # 3D: (1, 60, 20)
            input_reshaped = input_scaled.reshape(1, 60, len(self.FEATURE_COLS))
            pred_scaled = self._models['transformer'].predict(input_reshaped, verbose=0)

        pred_val = np.array(pred_scaled).reshape(-1, 1)
        rtt_real = self._scalers['y'].inverse_transform(pred_val)

        return float(rtt_real[0][0])

