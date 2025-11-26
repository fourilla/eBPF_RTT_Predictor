import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("--- [통합 평가] 데이터 및 모델 3종 로드 중... ---")

# ==========================================
# 1. 파일 로드
# ==========================================
try:
    # 데이터 로드 (Transformer는 LSTM과 동일한 X_test_lstm 사용)
    y_test_scaled = np.load('y_test.npy')
    X_test_lstm = np.load('X_test_lstm.npy')
    X_test_xgb = np.load('X_test_xgb.npy')

    # 스케일러 로드
    scaler_y = joblib.load('scaler_y.joblib')

    # 모델 로드 (compile=False: 예측 전용)
    model_lstm = load_model('model_lstm_best.keras', compile=False)
    model_xgb = joblib.load('model_xgboost_best.joblib')

    # [추가됨] Transformer 모델
    try:
        model_trans = load_model('model_transformer_best.keras', compile=False)
    except:
        print("[경고] Transformer 모델 파일이 없습니다. 해당 모델은 제외하고 진행합니다.")
        model_trans = None

except FileNotFoundError as e:
    print(f"[오류] 필수 파일이 없습니다: {e}")
    exit()

# ==========================================
# 2. 예측 수행 및 값 복원
# ==========================================
print("--- 예측 수행 중... ---")

# 1) LSTM
pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()

# 2) XGBoost
pred_xgb_scaled = model_xgb.predict(X_test_xgb).reshape(-1, 1)
pred_xgb = scaler_y.inverse_transform(pred_xgb_scaled).flatten()

# 3) Transformer (모델이 있을 경우만)
if model_trans:
    pred_trans_scaled = model_trans.predict(X_test_lstm, verbose=0)
    pred_trans = scaler_y.inverse_transform(pred_trans_scaled).flatten()
else:
    pred_trans = np.zeros_like(pred_lstm)  # 없으면 0으로 채움

# 실제값 복원
y_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# DataFrame 생성 (음수 보정 포함)
df = pd.DataFrame({'Actual': y_real, 'XGBoost': pred_xgb, 'LSTM': pred_lstm})
if model_trans:
    df['Transformer'] = pred_trans

# 물리적 보정 (RTT > 0)
df = df[df['Actual'] > 0]
for col in df.columns:
    if col != 'Actual':
        df[col] = df[col].apply(lambda x: max(x, 0))


# ==========================================
# 3. 성능 지표 계산 및 출력
# ==========================================
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)
    return [model_name, round(mae, 4), round(rmse, 4), round(mape, 2), round(r2, 4)]


results = []
results.append(calculate_metrics(df['Actual'], df['XGBoost'], 'XGBoost'))
results.append(calculate_metrics(df['Actual'], df['LSTM'], 'LSTM'))
if model_trans:
    results.append(calculate_metrics(df['Actual'], df['Transformer'], 'Transformer'))

metrics_df = pd.DataFrame(results, columns=['Model', 'MAE (ms)', 'RMSE (ms)', 'MAPE (%)', 'R2 Score'])

print("\n" + "=" * 50)
print(" 📊 모델 3종 성능 최종 성적표")
print("=" * 50)
print(metrics_df.to_string(index=False))
print("-" * 50)

# ==========================================
# 4. 그래프 그리기 (3종 포함)
# ==========================================
print("--- 그래프 생성 및 저장 중 ---")

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

# [Graph 1] 전체 흐름 (Sampling)
SAMPLE_RATE = 50
ax[0].plot(df['Actual'].values[::SAMPLE_RATE], label='Actual', color='black', alpha=0.4, linewidth=1)
ax[0].plot(df['XGBoost'].values[::SAMPLE_RATE], label='XGBoost', color='red', linestyle='--', linewidth=1.5)
ax[0].plot(df['LSTM'].values[::SAMPLE_RATE], label='LSTM', color='blue', alpha=0.6, linewidth=1)
if model_trans:
    # Transformer는 녹색으로 표시
    ax[0].plot(df['Transformer'].values[::SAMPLE_RATE], label='Transformer', color='green', alpha=0.7, linewidth=1)

ax[0].set_title(f'1. Overall Trend (Downsampled by {SAMPLE_RATE})', fontsize=14, fontweight='bold')
ax[0].set_ylabel('RTT (ms)')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# [Graph 2] 상세 확대 (Zoom In)
START = 100
END = 400
ax[1].plot(df['Actual'].values[START:END], label='Actual', color='black', linewidth=2)
ax[1].plot(df['XGBoost'].values[START:END], label='XGBoost', color='red', linestyle='--', linewidth=2)
ax[1].plot(df['LSTM'].values[START:END], label='LSTM', color='blue', alpha=0.6, linewidth=2)
if model_trans:
    ax[1].plot(df['Transformer'].values[START:END], label='Transformer', color='green', alpha=0.8, linewidth=2)

ax[1].set_title(f'2. Detailed View (Zoom in: Step {START} to {END})', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('RTT (ms)')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_3_models.png', dpi=300)
print("그래프 저장 완료: comparison_3_models.png")