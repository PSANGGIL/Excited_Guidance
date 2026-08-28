codex resume 01a041d0-db57-7c53-b974-ec88fac4192b

# Codex 작업 로그

이 파일은 세션이 바뀌어도 작업 맥락을 복구할 수 있도록 주요 내용을 누적 기록한다.

## 기록 원칙

- 주요 분석 결과와 그 근거
- 사용자와 합의한 결정 및 우선순위
- 코드·설정 변경 내용과 검증 결과
- 장시간 학습·예측 작업의 실행 상태와 최종 결과
- 남은 문제와 다음 작업
- 단순 조회, 중간 진행 출력, 중요하지 않은 시행착오는 생략

## 2026-08-27 — 모델 개선 작업 맥락 복구

### 사용자 요청

- 앞으로 주요 작업 내용을 프로젝트 루트의 `codex_log.md`에 자동으로 기록한다.

### 직전 학습 결과

- 실행 경로: `runs/os_cfg_20260825_111949`
- 학습은 epoch 634에서 조기 종료됐다.
- 최적 체크포인트: epoch 584, step 236339
- `val_cond_total_loss`: 1.593816
- `val_atom_rms_bar`: 1.167157
- `val_bond_acc_bar`: 0.990177
- `val_e_bonded_precision`: 0.946949
- `val_e_bonded_recall`: 0.691286
- `val_e_bonded_f1`: 0.798125
- `val_e_macro_f1`: 0.778531
- single/double/triple/aromatic F1: 0.778803 / 0.629882 / 0.713162 / 0.775551
- double-bond recall: 0.495320
- 분자 원자가 통과율: 0.445600

### 분석

- 전체 bond accuracy 0.99는 no-bond 클래스 비중 때문에 실제 결합 성능을 과대평가한다.
- 결합 정밀도는 높지만 재현율이 낮으며, 특히 double bond 재현율이 병목이다.
- 원자가 손실만 강화하면 결합을 적게 예측해 재현율이 더 악화될 수 있다.
- 다음 모델 개선 후보는 상한이 있는 결합 클래스 불균형 가중치다.

### 실행 상태

- best checkpoint 기반 10,000개 생성 작업이 시작됐다.
- 확인 당시 실제 Python 자식 프로세스가 실행 중이었으며 기존 작업은 중단하지 않았다.

### 미완료 작업

- 결합 클래스 불균형 보정 패치는 샌드박스 편집기 오류로 적용되지 않았다.
- 모델 파일과 테스트에는 해당 개선으로 인한 변경이 없다.

## 2026-08-27 — 조건 반영 손실 개선

- condition gain 0.79%를 조건 무시의 핵심 근거로 확인했다.
- 브랜치 codex/condition-margin-loss를 생성했다.
- 동일 noise/time에서 올바른 조건과 shuffled 조건을 비교하는 hinge margin loss를 추가했다.
- 첫 설정은 p_uncond=0, margin 0.05, weight 0.5다.
- 조건 반응성을 확인한 뒤 CFG를 다시 활성화한다.

codex resume 01a041d0-db57-7c53-b974-ec88fac4192b
codex resume 01a041d0-db57-7c53-b974-ec88fac4192b

## 2026-08-27 — 기존 best checkpoint 10,000개 생성 결과

- checkpoint: runs/os_cfg_20260825_111949/checkpoints/epoch=584-step=236340.ckpt
- 출력: generated/20260827_060226, target f_osc=0.5, timesteps 250, seed 42
- 10,000개 생성 완료, valid/SDF 75개로 validity 0.75%
- valid 75개는 모두 unique·novel이며 데이터셋 중복은 0개
- 주요 실패: explicit valence 6,516회, kekulization 3,256회
- validation valence pass 44.6%와 실제 validity 0.75%의 괴리가 커서 현재 denoising validation이 iterative sampling 실패를 충분히 대표하지 못함
- 75개 모두 xTB/sTDA 계산 성공
- 계산 f_osc 평균 0.5843, 중앙값 0.4927, 범위 0.0277–1.9987, target MAE 0.3540
- 오차 0.05 이내 7/75(9.33%), 0.1 이내 10/75(13.33%), 0.2 이내 22/75(29.33%)
- 단일 target 결과라 condition 반영을 입증할 수 없으며 동일 seed의 다중 target response slope·상관 비교가 필요
- condition collapse와 별개로 생성 화학 유효성도 심각한 병목임을 확인

## 2026-08-28 — 생성 결과 상세 분석

- heavy-atom-only 전환은 보류하고 기존 생성 결과 분석을 우선함
- invalid 9,925개: explicit valence 6,516, kekulization 3,385, disconnected 24
- valence 오류 원소: C 2,866, N 1,753, H 1,044, O 722, 기타 131
- 주요 오류 조합: C5 1,790, N4 1,457, C6 1,026, H2 857
- 수소 오류만 제거해도 전체 invalid 문제는 해결되지 않으며 C/N/방향족 오류가 더 큼
- 생성 원자 수 중앙값 92, valid 원자 수 중앙값 68
- 크기별 validity: 30–49 5.59%, 50–69 3.15%, 70–89 0.86%, 90–109 0.11%, 110 이상 0%
- valid 구조는 heavy atom 중앙값 46, MW 중앙값 609.7, LogP 중앙값 6.96, QED 중앙값 0.179
- valid 구조도 크고 소수성이 높으며 고리 중앙값 8, 방향족 고리 중앙값 6으로 편향됨
- SDF 재독해 결과 75개 중 70개 sanitize 성공, 5개(1099,1471,4572,5442,5844)는 후속 계산 후 kekulization 실패
- 결합 길이 중앙값: aromatic 1.416 A, single 1.114 A(H 결합 포함), double 1.354 A, triple 1.136 A
- f_osc와 크기·MW·LogP·고리 수의 Pearson 상관은 모두 절댓값 0.15 이하로 단일 target 표본에서 뚜렷한 구조-물성 관계가 없음
- 가장 가까운 생성 index는 6078(f_osc 0.4927, 오차 0.0073), 2244(0.5143), 4572(0.4848)
- 핵심 병목은 (1) 큰 분자에서 누적되는 valence/aromatic 오류, (2) valid 표본의 크고 다환 방향족 편향, (3) 단일 target으로 조건 반응성 판정 불가

## 2026-08-28 — xTB/sTDA top-3 OS annotated PNG

- heavy-atom-only 전환은 계속 보류함
- tracked 실행파일 generated/xtb_stda_run에 --render-only 옵션 추가
- 기존 top-1 CSV는 유지하고 state 1–10 중 OS top-3 long-format CSV를 추가
- 각 분자 아래 ID와 #1–#3 state/OS를 표시한 전체 grid 및 20개 단위 page PNG 추가
- canonical SMILES를 우선 사용해 xTB 이후 SDF sanitize 실패 구조도 그림 생성 가능
- 기존 75개 결과로 검증: top-3 CSV 225행, 전체 PNG 1개, page PNG 4개 생성 성공
- 출력: generated/20260827_060226/sdf/stda_top3_os_state1-10.csv 및 stda_top3_os_grid*.png

## 2026-08-28 — top-3 OS PNG 가독성 개선

- 기존 4열 450x360 RDKit legend 방식은 구조와 텍스트가 작고 겹쳐 폐기
- 구조 900x560 + 독립 정보 패널 900x180으로 분리
- ID 32px bold, top-3 state/OS 28px, 2열 x 4행 페이지 구성
- 분자별 900x740 개별 PNG 75개도 추가
- 대표 00105.png를 실제 열어 구조·ID·top-3 텍스트 가독성 확인 완료

## 2026-08-28 — 기존 학습 condition gain 수렴 분석

- 분석 지표: val_condition_gain_percent의 50-epoch trailing mean
- 최신 run(20260825, epoch 0–634): 초기 50 epoch 평균 0.168%, epoch 300 0.500%, 350 0.566%, 450 0.590%, 550 0.614%, 최종 0.605%
- 안정 plateau 시작은 약 epoch 349(최종 평균 대비 +/-0.075%p 기준)
- epoch 500–634 평균 0.611%, 기울기 -0.0014%p/100 epoch로 사실상 완전 수렴
- best-loss checkpoint epoch 584의 단일 gain은 0.788%지만 trailing mean 0.619%로 plateau 범위
- 최대 1.246%(epoch 425)는 지속되지 않는 단발성 spike이며 수렴 수준으로 해석하면 안 됨
- 이전 run(20260823)도 마지막 50 epoch 평균 0.624%로 동일한 약 0.6% ceiling에 도달
- 두 run이 서로 다른 학습 길이에도 같은 ceiling을 보여 현 loss/condition injection 구조의 한계일 가능성이 높음
- 결론: 기존 방식은 epoch 350 전후 실질 수렴, 500 이후 추가 학습 무의미, 조건 반영률 ceiling 약 0.6%

## 2026-08-28 — 조건 반영률 개선 방법 조사

- 현재 조건은 node scalar 초기 embedding에 concatenate_sum으로 한 번만 주입됨
- 기존 sampling guide weight는 x=2, a=c=e=1이라 categorical 원자/전하/결합에는 CFG extrapolation이 없음
- 기존 두 run의 condition gain ceiling 약 0.6%는 학습 시간보다 objective/injection 한계 신호
- 최우선 평가 오류: predict.py는 f_osc를 S1 oscillator strength로 정의하지만 기존 sTDA 평가는 state 1–10 최대 OS를 target과 비교함. 조건 평가는 반드시 state 1 OS로 재계산해야 함
- 1순위 진단: clean molecule에서 f_osc를 예측하는 별도 surrogate의 validation R2/MAE로 데이터 내 구조-물성 신호 상한 확인
- 1순위 학습 개선: property label을 입력받지 않는 frozen/pretrained noisy-state property predictor를 학습해 auxiliary loss 및 classifier guidance에 사용
- 2순위 구조 개선: condition을 초기 1회가 아니라 각 GNN/GVP block에 FiLM(scale+shift) 방식으로 주입
- 3순위 margin 개선: 단순 batch roll 대신 property 차이가 큰 hard negative를 선택하고 margin을 |delta y|에 비례시킴
- 가장 corrupted된 timestep 구간에서 condition margin/property loss 가중치를 높여 조건이 필요한 구간에 집중
- conditional-unconditional 분리가 확인된 뒤 p_uncond를 복원하고 a/c/e CFG weight도 1 초과 grid 탐색
- 권장 단계: S1 평가 수정 -> surrogate predictability -> margin pilot -> per-block FiLM + frozen property loss -> categorical CFG sweep

## 2026-08-28 — Frozen property predictor 사용 재검토

- 사용자 지적대로 별도 OS predictor를 generator loss에 붙이면 surrogate 정확도/calibration 문제와 reward hacking 위험이 추가됨
- f_osc는 양자화학 물성이므로 일반 descriptor보다 surrogate 편향 위험이 큼
- 우선순위 수정: 1차 실험은 predictor 없이 per-block FiLM + molecule별 hard-negative margin만 사용
- 실제 조건 성능 판정은 predictor가 아니라 xTB/sTDA의 S1 OS로 수행
- predictor는 독립 validation 성능과 uncertainty가 충분히 검증된 경우에만 보조 guidance/reranking 용도로 제한 검토
- predictor 사용 시에도 ensemble/uncertainty 필터와 실제 sTDA 최종 검증이 필수


## 2026-08-28 — Per-block FiLM + molecule-wise hard-negative margin 구현

- 새 브랜치 `codex/film-hard-negative-conditioning` 생성
- frozen OS predictor 없이 조건 반영 구조만 개선
- property embedding을 모든 GVP convolution 뒤에 FiLM으로 재주입
- scalar feature에는 scale+shift, vector channel에는 invariant scale만 적용해 E(3) 등변성 보존
- FiLM 마지막 projection은 zero initialization해 초기 동작을 identity로 설정
- 기존 batch-roll/aggregate margin을 분자별 denoising loss 기반 hard-negative margin으로 교체
- 각 분자는 배치 내 normalized property 거리가 가장 먼 target을 negative로 선택
- 요구 margin은 property distance에 비례하며 `condition_margin_distance_cap`으로 상한 적용
- correct/negative pass는 conditional mode와 RNG 복원으로 같은 time/corruption을 공유
- 진단 metric 추가: negative distance, active margin fraction, correct-vs-negative loss gain
- 검증: 관련 unittest 15개 통과, config.example 모델 구성 통과
- 실제 데이터 2분자 hard-negative training forward/backward 통과: 모든 loss finite, FiLM gradient 연결 확인
- 전체 configured batch CPU forward는 메모리 한계(code 137)로 완료하지 못했으며 GPU 검증 대상
