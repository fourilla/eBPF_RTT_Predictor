import pandas as pd
import numpy as np
import glob
import os


def augment_data(input_pattern="*.csv", output_dir="augmented_data", expand_factor=10, noise_level=0.02,
                 num_variations=5):
    """
    CSV 파일을 읽어서 보간(Interpolation) 후, 서로 다른 랜덤 노이즈를 섞어 여러 개의 변형 데이터를 생성합니다.

    :param input_pattern: 대상 파일 패턴 (예: "*.csv")
    :param output_dir: 저장할 폴더
    :param expand_factor: 데이터를 몇 배로 늘릴지 (예: 10 -> 1초 간격을 0.1초로 쪼갬)
    :param noise_level: 노이즈 강도 (0.05 -> 5% 변동)
    :param num_variations: 파일 하나당 생성할 변형 개수 (예: 5 -> 파일 1개로 5개 생성)
    """

    # 저장 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob.glob(input_pattern))
    if not files:
        print("[오류] 변환할 CSV 파일이 없습니다.")
        return

    print(f"🚀 데이터 증강 시작")
    print(f"   - 대상 파일 수: {len(files)}개")
    print(f"   - 파일당 생성: {num_variations}개 (총 {len(files) * num_variations}개 파일 생성)")
    print(f"   - 데이터 확장: {expand_factor}배 (보간)")

    for f in files:
        try:
            # 1. 원본 데이터 로드
            df = pd.read_csv(f)
            base_name = os.path.splitext(os.path.basename(f))[0]

            # apply arr[i] = average of arr[i-4:i]
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].rolling(window=4, min_periods=1).mean()

            # 2. 보간(Interpolation)을 위한 공통 인덱스 생성
            # 원본 데이터 길이에서 expand_factor만큼 촘촘한 그리드를 만듦
            old_index = np.arange(len(df))
            new_index = np.linspace(0, len(df) - 1, len(df) * expand_factor)

            # [최적화] 베이스(보간된) 데이터를 먼저 만듭니다. (노이즈 없는 깨끗한 상태)
            base_data = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 선형 보간
                    base_data[col] = np.interp(new_index, old_index, df[col])
                else:
                    # 문자열 등은 가장 가까운 값 복제
                    indices = np.round(new_index).astype(int)
                    indices = np.clip(indices, 0, len(df) - 1)
                    base_data[col] = df[col].iloc[indices].values

            base_df = pd.DataFrame(base_data)

            # 3. N개의 변형(Variation) 생성
            for v in range(1, num_variations + 1):
                aug_df = base_df.copy()

                # 각 변형마다 서로 다른 노이즈 주입
                for col in aug_df.columns:
                    # Time이나 Label은 노이즈 제외
                    if 'time' in col.lower() or 'label' in col.lower():
                        if 'label' in col.lower():
                            aug_df[col] = np.round(aug_df[col])  # 라벨은 정수로
                        continue

                    # 노이즈 추가 (매번 랜덤하게 생성됨)
                    vals = aug_df[col].values
                    sigma = np.std(vals) * noise_level
                    if sigma == 0: sigma = 0.0001

                    # 랜덤 시드를 고정하지 않음 -> 루프 돌 때마다 다른 노이즈 생성
                    noise = np.random.normal(0, sigma, size=len(vals))
                    aug_df[col] = vals + noise
                    aug_df[col] = aug_df[col].rolling(window=3, min_periods=1, center=True).mean()
                    aug_df[col] = aug_df[col].clip(lower=0)

                # 4. 저장 (파일명에 v1, v2... 붙임)
                save_name = f"aug_v{v}_{base_name}.csv"
                save_path = os.path.join(output_dir, save_name)
                aug_df.to_csv(save_path, index=False)

            print(f" - [완료] {f} -> {num_variations}개 변형 생성 완료")

        except Exception as e:
            print(f" - [실패] {f}: {e}")

    print("\n✨ 모든 증강 작업이 완료되었습니다.")
    print(f"📂 결과 폴더: {output_dir}")


if __name__ == "__main__":

    augment_data(
        input_pattern="*.csv",
        output_dir="../data",
        expand_factor=10,  # 10배로 뻥튀기 (보간)
        noise_level=0.02,  # 5% 정도의 랜덤 노이즈
        num_variations=1  # 파일 하나당 5가지 버전 만들기
    )