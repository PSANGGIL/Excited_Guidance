import os
import glob
import re
import pandas as pd
import numpy as np

def parse_lowest_state(file_path):
    """
    단일 stda 로그 파일에서 진짜 결과 테이블을 찾아 
    가장 에너지가 낮은 상태(State 1)의 eV와 fL 값만 추출합니다.
    """
    # 파일명에서 분자 ID 추출 (예: '00015/00015_stda.log' -> '00015')
    mol_id = os.path.basename(file_path).replace('_stda.log', '')
    
    # 정규표현식: state, eV, nm, fL, Rv
    pattern = re.compile(r'^\s*(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([-\d\.]+)')
    
    parsed_data = []
    is_target_table = False  # 가짜 테이블(CSF)을 거르기 위한 안전 스위치
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 1. 단수형/복수형 오타를 모두 잡기 위해 공통 문자열로 스위치를 켬
                if "transition moments and TDA amplitudes" in line:
                    is_target_table = True
                    continue  # 제목 줄 자체는 건너뜀
                
                # 2. 스위치가 켜진 이후부터만 정규표현식으로 데이터 파싱
                if is_target_table:
                    match = pattern.search(line)
                    if match:
                        state = int(match.group(1))
                        
                        parsed_data.append({
                            'mol_id': mol_id,
                            'state': state,
                            'stda_eV': float(match.group(2)),
                            'stda_fL': float(match.group(4))
                        })
                        
                        # 가장 첫 번째 상태(State 1, Lowest State)를 찾았으므로 더 이상 읽지 않고 종료
                        if state == 1:
                            break
                            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return parsed_data

# =====================================================================
# 메인 실행부
# =====================================================================

# 1. 모든 폴더 내의 _stda.log 파일 검색
log_files = glob.glob('*/*_stda.log')
print(f"총 {len(log_files)}개의 stda 로그 파일을 찾았습니다.\n파싱을 시작합니다...\n")

# 2. 전체 로그 파일 파싱 및 하나의 DataFrame으로 병합
all_stda_data = []
for file_path in log_files:
    parsed = parse_lowest_state(file_path)
    if not parsed:
        # 계산이 도중에 터졌거나 정상적인 결과가 없는 경우 경고 출력
        print(f"⚠️ 경고: {file_path} 에서 최종 결과를 찾을 수 없습니다.")
    else:
        all_stda_data.extend(parsed)

df_stda_lowest = pd.DataFrame(all_stda_data)

# 데이터가 아예 없는 경우 방지
if df_stda_lowest.empty:
    print("\n❌ 파싱된 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

# 3. 생성 모델(MolGuidance)의 예측 결과 불러오기 
# ⚠️ 실제 사용하실 때는 아래 주석을 풀고 실제 CSV 경로를 입력하세요.
# df_model = pd.read_csv('your_model_output.csv')

# --- 테스트를 위한 가상의 모델 예측 데이터 생성 코드 ---
mock_model_data = []
for mol in df_stda_lowest['mol_id'].unique():
    mock_model_data.append({
        'mol_id': mol,
        'state': 1,
        'model_fL': np.random.uniform(0, 0.5) # 0~0.5 사이의 가상의 fL 예측값
    })
df_model = pd.DataFrame(mock_model_data)
# ---------------------------------------------------

# 4. 분자 ID(mol_id)를 기준으로 양쪽 데이터 병합 (교집합)
df_compare = pd.merge(df_stda_lowest, df_model, on=['mol_id', 'state'], how='inner')

# 5. 오차(Absolute Error) 계산
df_compare['error_fL'] = np.abs(df_compare['stda_fL'] - df_compare['model_fL'])

# 6. 결과 출력 및 검증
print("\n=== Lowest State (S1) 비교 결과 요약 (상위 10개) ===")
print(df_compare[['mol_id', 'stda_eV', 'stda_fL', 'model_fL', 'error_fL']].head(10))

# 전체 MAE(평균 절대 오차) 계산
mae_total = df_compare['error_fL'].mean()
print(f"\nS1 fL Mean Absolute Error (MAE): {mae_total:.6f}")

# 7. 최종 결과를 CSV로 저장
output_filename = 's1_stda_vs_model_comparison.csv'
df_compare.to_csv(output_filename, index=False)
print(f"\n정상 파싱된 전체 비교 결과가 '{output_filename}' 파일로 저장되었습니다.")
