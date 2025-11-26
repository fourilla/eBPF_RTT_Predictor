import numpy as np
import joblib
import os
import gc

# 모델 라이브러리
from xgboost import XGBRegressor
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------------------------------
# [내부 함수] 트랜스포머 모델 빌더
# -------------------------------------------------------
def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, mlp_units=[64],
                            dropout=0.1, mlp_dropout=0.1):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 트랜스포머 인코더 블록 반복
    for _ in range(num_transformer_blocks):
        # 1. Multi-Head Attention
        x_att = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Add()([x, x_att])  # Residual Connection (입력 + 어텐션 결과)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # 2. Feed Forward Network (Conv1D 사용)
        x_ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x_ff = layers.Dropout(dropout)(x_ff)
        x_ff = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x_ff)

        x = layers.Add()([x, x_ff])  # Residual Connection
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 출력 헤드 (Regression)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1)(x)  # RTT 예측 (1개의 값)

    return Model(inputs, outputs)

def train_data(model_name):
    model_name = model_name.lower()
    print(f"\n{'=' * 40}")
    print(f"🚀 [{model_name.upper()}] 학습 프로세스 시작")
    print(f"{'=' * 40}")

    # -----------------------------------------
    # 1. XGBoost 학습 로직
    # -----------------------------------------

    if model_name == "xgboost":
        print(f"\n{'=' * 40}")
        print(f"🔎 [XGBoost] GridSearchCV 파라미터 튜닝 시작")
        print(f"{'=' * 40}")

        # 1. 데이터 로드 (메모리 안전 모드)
        try:
            print(" - [1/4] 데이터 로드 및 경량화(float32) 중...")
            X_train = np.load('X_train_xgb.npy').astype(np.float32)
            y_train = np.load('y_train.npy').astype(np.float32)
            X_test = np.load('X_test_xgb.npy').astype(np.float32)
            y_test = np.load('y_test.npy').astype(np.float32)
            print(f"   -> 로드 완료: {X_train.shape}")
        except FileNotFoundError:
            print(f"[오류] 데이터 파일이 없습니다.")
            return

        # 2. 파라미터 그리드 설정
        # (범위를 너무 넓게 잡으면 시간이 오래 걸리므로 핵심 구간 위주로 설정)
        param_grid = {
            'n_estimators': [100, 300, 500],  # 트리 개수
            'max_depth': [4, 6, 8],  # 트리 깊이
            'learning_rate': [0.05, 0.1],  # 학습률
            'subsample': [0.8, 1.0]  # 데이터 샘플링 비율
        }

        print(f" - [2/4] 탐색할 파라미터 조합 설정 완료")
        print(f"   -> {param_grid}")

        # 3. GridSearchCV 설정 및 학습
        # 주의: n_jobs=-1을 GridCV에 주면 데이터 복사본이 생겨 메모리 터질 수 있음.
        # 따라서 모델(xgb)에만 n_jobs=-1을 주고, GridCV는 순차적(n_jobs=1)으로 돌리는 게 안전함.
        xgb = XGBRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # 회귀이므로 MSE 기준 (음수값 사용)
            cv=3,  # 3-Fold 교차 검증
            verbose=2,  # 진행상황 출력
            n_jobs=1  # 메모리 안전을 위해 1로 설정 (병렬처리는 xgb가 수행)
        )

        print(" - [3/4] Grid Search 학습 시작 (시간이 다소 소요됩니다)...")
        grid_search.fit(X_train, y_train)

        # 4. 최적 결과 도출 및 메모리 정리
        best_xgb = grid_search.best_estimator_
        print(f"\n✨ 최적 파라미터 발견: {grid_search.best_params_}")

        # 학습 데이터 메모리 해제
        del X_train, y_train
        gc.collect()
        print("   -> 학습 데이터 메모리 해제 완료")

        # 5. 최종 평가 및 저장
        print(" - [4/4] 최종 모델 평가 및 저장 중...")
        y_pred = best_xgb.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 [GridCV 최종 결과] MSE: {mse:.4f}, R2: {r2:.4f}")

        # 모델 저장
        joblib.dump(best_xgb, 'model_xgboost_best.joblib')
        print("💾 최적 모델 저장 완료: model_xgboost_best.joblib")

    # -----------------------------------------
    # 2. LSTM 학습 로직
    # -----------------------------------------
    elif model_name == "lstm":
        # 데이터 로드
        try:
            X_train = np.load('X_train_lstm.npy')
            y_train = np.load('y_train.npy')
            X_test = np.load('X_test_lstm.npy')
            y_test = np.load('y_test.npy')
        except FileNotFoundError:
            print(f"[오류] 데이터 파일이 없습니다. process_data.py를 먼저 실행하세요.")
            return

        print(f" - 데이터 로드 완료: {X_train.shape}")

        # 모델 구성
        model = Sequential()
        # 입력 형태 자동 인식 (TimeSteps, Features)
        model.add(LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # 회귀 출력을 위한 노드 1개

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        # 콜백 설정 (과적합 방지 및 모델 저장)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('model_lstm_best.keras', monitor='val_loss', save_best_only=True)
        ]

        print(" - 학습 시작 (Early Stopping 적용)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,  # 최대 50번 돌되
            batch_size=64,
            callbacks=callbacks,  # 성능 안 오르면 조기 종료
            verbose=1
        )

        # 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 [LSTM 결과] MSE: {mse:.4f}, R2: {r2:.4f}")
        print("💾 모델 저장 완료: model_lstm_best.keras")

        # ==========================================
        # 3. Transformer 학습 (신규 추가)
        # ==========================================
    elif model_name == "transformer":
        # 데이터 로드 (LSTM과 동일한 3D 데이터 사용)
        try:
            print(" - 데이터 로드 중...")
            X_train = np.load('X_train_lstm.npy')
            y_train = np.load('y_train.npy')
            X_test = np.load('X_test_lstm.npy')
            y_test = np.load('y_test.npy')
        except FileNotFoundError:
            print(f"[오류] 데이터 파일(.npy)이 없습니다.")
            return

        print(f" - 데이터 형태: {X_train.shape}")  # (Samples, 60, 20)

        # 모델 생성
        input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 20)

        model = build_transformer_model(
            input_shape,
            head_size=64,  # 어텐션 헤드 크기
            num_heads=4,  # 어텐션 헤드 개수 (병렬 처리)
            ff_dim=64,  # 내부 피드포워드 망 크기
            num_transformer_blocks=2,  # 인코더 블록 층 수 (너무 깊으면 학습 어려움)
            mlp_units=[128],  # 최종 출력층 전의 Dense Layer
            dropout=0.1,
            mlp_dropout=0.1
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('model_transformer_best.keras', monitor='val_loss', save_best_only=True)
        ]

        print(" - 트랜스포머 학습 시작...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,  # 필요시 늘리세요
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 [Transformer 결과] MSE: {mse:.4f}, R2: {r2:.4f}")
        print("💾 모델 저장 완료: model_transformer_best.keras")

    else:
        print(f"[오류] 알 수 없는 모델입니다: {model_name}.")