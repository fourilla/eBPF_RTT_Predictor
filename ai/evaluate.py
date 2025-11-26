import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("--- [성능 평가] 데이터 및 모델 로드 중... ---")

# ==========================================
# 1. 파일 로드
# ==========================================
try:
    # 테스트 데이터 로드
    # (Transformer는 LSTM과 동일한 3D 입력(X_test_lstm)을 사용합니다)
    y_test_scaled = np.load('y_test.npy')
    X_test_lstm = np.load('X_test_lstm.npy')
    X_test_xgb = np.load('X_test_xgb.npy')

    # 스케일러 로드
    scaler_y = joblib.load('scaler_y.joblib')

    # 모델 로드 (compile=False: 예측 전용 로드)
    model_lstm = load_model('model_lstm_best.keras', compile=False)
    model_xgb = joblib.load('model_xgboost_best.joblib')

    # [추가됨] Transformer 모델 로드
    model_transformer = load_model('model_transformer_best.keras', compile=False)

except FileNotFoundError as e:
    print(f"[오류] 필요 파일이 없습니다: {e}")
    print("팁: 'model_transformer_best.keras'가 없다면 Transformer 학습을 먼저 실행하세요.")
    exit()

# ==========================================
# 2. 예측 수행 및 값 복원 (Inverse Transform)
# ==========================================
print("--- 예측 수행 중... ---")

# 1) LSTM 예측
pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()

# 2) XGBoost 예측
pred_xgb_scaled = model_xgb.predict(X_test_xgb).reshape(-1, 1)
pred_xgb = scaler_y.inverse_transform(pred_xgb_scaled).flatten()

# 3) [추가됨] Transformer 예측 (입력은 LSTM과 동일)
pred_trans_scaled = model_transformer.predict(X_test_lstm, verbose=0)
pred_trans = scaler_y.inverse_transform(pred_trans_scaled).flatten()

# 4) 실제값 복원
y_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# DataFrame으로 합치기
df = pd.DataFrame({
    'Actual': y_real,
    'XGBoost': pred_xgb,
    'LSTM': pred_lstm,
    'Transformer': pred_trans
})

# [옵션] 음수 값 보정 (RTT는 물리적으로 0 이상이어야 함)
df = df[df['Actual'] > 0]
for col in ['XGBoost', 'LSTM', 'Transformer']:
    df[col] = df[col].apply(lambda x: max(x, 0))


# ==========================================
# 3. 성능 지표 계산 함수
# ==========================================
def calculate_metrics(y_true, y_pred, model_name):
    # MAE (평균 절대 오차): 낮을수록 좋음
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE (평균 제곱근 오차): 튀는 값(Outlier)에 민감. 낮을수록 좋음
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (평균 절대 비율 오차): 퍼센트 오차. 낮을수록 좋음
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # R2 Score (결정 계수): 1에 가까울수록 좋음
    r2 = r2_score(y_true, y_pred)

    return [model_name, round(mae, 4), round(rmse, 4), round(mape, 2), round(r2, 4)]


# ==========================================
# 4. 결과 집계 및 출력
# ==========================================
results = []
results.append(calculate_metrics(df['Actual'], df['XGBoost'], 'XGBoost'))
results.append(calculate_metrics(df['Actual'], df['LSTM'], 'LSTM'))
results.append(calculate_metrics(df['Actual'], df['Transformer'], 'Transformer'))

metrics_df = pd.DataFrame(results, columns=['Model', 'MAE (ms)', 'RMSE (ms)', 'MAPE (%)', 'R2 Score'])

print("\n" + "=" * 50)
print(" 📊 모델 3종 성능 최종 성적표")
print("=" * 50)
print(metrics_df.to_string(index=False))
print("-" * 50)

# 승자 판별 (R2 Score 기준)
best_model_row = metrics_df.loc[metrics_df['R2 Score'].idxmax()]
best_model_name = best_model_row['Model']
best_r2 = best_model_row['R2 Score']

print(f"🏆 [최종 승자] {best_model_name} (R2: {best_r2})")

# (추가 분석) Transformer vs XGBoost 비교
xgb_rmse = metrics_df[metrics_df['Model'] == 'XGBoost']['RMSE (ms)'].values[0]
trans_rmse = metrics_df[metrics_df['Model'] == 'Transformer']['RMSE (ms)'].values[0]

print("\n[분석 코멘트]")
if xgb_rmse < trans_rmse:
    diff = trans_rmse - xgb_rmse
    print(f"- XGBoost가 Transformer보다 RMSE가 {diff:.2f}ms 더 낮습니다.")
    print("- 이는 정형 데이터에서 트리 모델이 딥러닝보다 효율적임을 보여줍니다.")
else:
    diff = xgb_rmse - trans_rmse
    print(f"- Transformer가 XGBoost보다 RMSE가 {diff:.2f}ms 더 낮습니다.")
    print("- 시계열의 전역적 패턴 학습이 효과를 발휘했습니다.")