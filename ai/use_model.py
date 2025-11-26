from train_data import train_data

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

class rtt_predictor:
    _models = {}
    _scalers = {}

    # 피처 순서 (학습 데이터와 100% 일치해야 함)
    FEATURE_COLS = [
        'RTT_ms_mean', 'packet_loss_rate_pct', 'retrans_segs_per_sec',
        'conn_fail_rate_server', 'listen_drops_per_sec',
        'Recv-Q_Bytes_max', 'Send-Q_Bytes_max', 'run_queue_len_cpu0',
        'cwnd_min', 'ssthresh_min', 'nic_usage_pct', 'total_packets_per_sec',
        'used_MB', 'available_MB', 'total_mem_events_per_sec',
        'cpu_throttle_per_sec', 'softirq_net_rx_per_sec', 'softirq_net_tx_per_sec',
        'iostat_r_await', 'iostat_aqu_sz'
    ]

    def __init__(self, initial_data):
        """
        :param initial_data: 초기 버퍼를 채울 데이터 (딕셔너리 1개).
        """
        # 리스트 안에 딕셔너리 60개를 채움 (초기화)
        self.sixty_sec_data = [initial_data.copy() for _ in range(60)]

        # 인스턴스 생성 시 모델 로드 시도
        self._load_artifacts()

    @classmethod
    def _load_artifacts(cls):
        # 이미 로드되어 있으면 스킵
        if cls._models and cls._scalers:
            return True

        print("--- [System] 모델 및 스케일러 로드 중... ---")

        # 1. 스케일러 로드
        if not cls._scalers:
            try:
                cls._scalers['X'] = joblib.load('scaler_X.joblib')
                cls._scalers['y'] = joblib.load('scaler_y.joblib')
            except FileNotFoundError:
                print("[오류] 스케일러 파일(.joblib)이 없습니다.")
                return False

        # 2. 모델 로드
        if not cls._models:
            # XGBoost
            try:
                cls._models['xgboost'] = joblib.load('model_xgboost_best.joblib')
            except:
                print("[주의] XGBoost 모델 파일이 없습니다.")

            # LSTM
            try:
                cls._models['lstm'] = load_model('model_lstm_best.keras', compile=False)
            except:
                print("[주의] LSTM 모델 파일이 없습니다.")

            # Transformer
            try:
                cls._models['transformer'] = load_model('model_transformer_best.keras', compile=False)
            except:
                print("[주의] Transformer 모델 파일이 없습니다.")

        return True

    """
    @param dict data     새로 들어온 데이터 1개 (Dictionary)
    @param str model     "lstm", "xgboost", "transformer" 중 선택

    @return float 예측된 RTT 값
    """

    def predict_data(self, data, model: str = "xgboost") -> float:
        # 1. 모델/스케일러 로드 확인
        if not self._models or not self._scalers:
            if not self._load_artifacts():
                return None

        # 2. 슬라이딩 윈도우 (맨 앞 제거, 맨 뒤 추가)
        self.sixty_sec_data.pop(0)
        self.sixty_sec_data.append(data)

        model = model.lower()
        if model not in self._models:
            print(f"[오류] '{model}' 모델이 로드되지 않았거나 파일이 없습니다.")
            return None

        # 3. 데이터 프레임 변환
        df = pd.DataFrame(self.sixty_sec_data)

        # 필수 컬럼 확인
        try:
            df = df[self.FEATURE_COLS]
        except KeyError as e:
            print(f"[오류] 입력 데이터에 필수 컬럼이 누락되었습니다: {e}")
            return None

        # 4. [수정됨] 정규화 (Scaling) 수행
        # 기존 코드에서 input_scaled = 0.0 으로 되어있던 부분을 수정함
        try:
            input_scaled = self._scalers['X'].transform(df.values)
        except Exception as e:
            print(f"[오류] 스케일링 중 에러 발생: {e}")
            return None

        # 5. 모델별 입력 형태 변환 및 예측
        pred_scaled = 0.0

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

        # 6. 역변환 (Inverse Transform)
        # 결과가 스칼라일 수도, 배열일 수도 있어서 안전하게 처리
        pred_val = np.array(pred_scaled).reshape(-1, 1)
        rtt_real = self._scalers['y'].inverse_transform(pred_val)

        return float(rtt_real[0][0])


def fast_train():
    # train_data("xgboost")
    train_data("lstm")
    train_data("transformer")

if __name__ == '__main__':
    fast_train()