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


## 2026-08-29 — FiLM hard-negative 학습 시작

- 최초 `max_num_edges=2,000,000` 실행은 hard-negative 이중 forward가 GPU 174.8/178.3 GiB를 사용해 첫 batch에서 OOM
- `max_num_edges=1,000,000` 첫 재시도에서 condition diagnostic의 CPU/GPU device mismatch 발견
- `negative_distance`를 loss device로 이동하도록 수정하고 commit `76deacf` 생성
- 최종 run: `runs/os_cfg_film_hardneg_e1m_fix_20260829_043256`
- 로그: `logs/train_film_hardneg_e1m_fix_20260829.log`
- fresh start, seed 42, W&B disabled, train edge budget 1,000,000
- sanity validation 통과 후 epoch 0 학습 진행 확인
- GPU 약 96,022 MiB/183,359 MiB, utilization 82%; 초기 60/736 step까지 정상 진행


## 2026-08-30 — Condition negative curriculum 및 10% GPU smoke profile

- 브랜치 `codex/condition-negative-curriculum`에서 far→medium→near negative curriculum 구현
- curriculum 커밋 `c053539`을 원격 브랜치에 푸시
- 별도 smoke launcher `run_curriculum_10pct_smoke.sh` 추가
- 테스트 설정: batch size 128, edge budget 220,000, property embedding 128
- vector field: scalar 128, edge 64, vector 8, molecule updates 3
- 기존 full run 약 96,508 MiB 위에서 총 peak 113,596 MiB 측정
- 테스트 프로세스 추가 peak 약 17,088 MiB로 B200 전체 183,359 MiB의 약 9.3%
- 10 train batches와 validation, checkpoint 저장까지 정상 완료
- 장기 학습 launcher `run_curriculum_10pct_train.sh` 추가
- 장기 run: `runs/os_cfg_curriculum_10pct_train_20260830_063839`
- 로그: `logs/train_curriculum_10pct_20260830.log`
- `setsid -f nohup`으로 분리 실행, PID 3558362
- 실제 장기 batch에서 약 19,118 MiB 사용(전체 B200의 약 10.4%), epoch 0 정상 진행


## 2026-09-02 — Kekulé bond representation 분리 구현

- 새 브랜치 `codex/kekule-bond-representation` 생성
- 모델이 aromatic을 독립 class로 생성하지 않고 `none/single/double/triple` 4개 class만 학습하도록 설정
- 원본 `data_`는 보존하고 별도 `data_kekule` 디렉터리를 만드는 `prepare_kekule_dataset.py` 추가
- RDKit Kekulization으로 aromatic bond를 원래 atom order에 정렬된 single/double 표현으로 변환하며 좌표·원자·전하·조건 property는 변경하지 않음
- 변환 후 train edge marginal을 4개 class 기준으로 다시 계산
- processed-data metadata에 `bond_representation: kekule`을 저장하고 config와 불일치하면 loader가 즉시 중단하도록 검증 추가
- CTMC mask index를 하드코딩된 5가 아니라 실제 마지막 class로 처리
- valence loss와 bond validation metric이 기존 5-class와 새 4-class를 모두 지원하도록 변경
- 생성 결과의 single/double Kekulé graph는 RDKit sanitization에서 aromaticity를 다시 인식
- synthetic benzene/기존 Kekulé/marginal 테스트 3개 통과
- 실제 train 250개 스모크 변환 통과: aromatic label 14,484개가 모두 single/double로 변환되고 label 4는 0개
- 전체 split을 별도 `data_kekule` 디렉터리로 변환 완료; 원본 `data_`는 유지
- loader 실검증: bond label 범위 1~3, edge feature 4차원, 새 p_e 합 1.0
- 주의: atom permutation까지 포함한 SMILES canonicalization은 좌표 정렬을 깨뜨리므로 하지 않으며, 고정된 원본 atom order에서 RDKit의 결정적 Kekulé assignment를 사용


## 2026-09-03 — 기존 5-class full-size best epoch 296 생성 재평가

- 체크포인트 `epoch=296-step=218592.ckpt`, target f_osc 0.5, seed 45, all-guidance 1.0
- 1,000개를 max batch size 32로 생성; 생성 프로세스 VRAM 약 4.3 GiB, OOM 없음
- RDKit molecule construction 1,000/1,000, unresolved mask 0
- sanitize 성공 4/1,000(0.4%), 그중 single-component/SMILES/SDF 성공 2/1,000(0.2%)
- 실패 분류: explicit valence 801, Kekulize 195, sanitize 후 multifragment 2
- 최종 2개는 모두 unique 및 train/val/test에 없는 novel molecule
- epoch 144 결과(4/1,000 valid)와 비교해 validation loss 개선이 최종 화학적 validity 증가로 이어지지는 않음

### 생성 성공 분자 구조 진단

- ID 609: C30H17FN8O2S, MW 572.585, logP 6.362, TPSA 147.21, QED 0.155, 8 rings(방향족 7), formal charge/radical 0
- ID 723: C28H17N5O3S, MW 503.543, logP 6.804, TPSA 109.69, QED 0.197, 방향족 ring 7, formal charge/radical 0
- 두 구조 모두 비결합 원자 충돌은 없지만 raw 좌표의 일부 aromatic bond가 비정상적으로 김: ID 609 최대 1.646 Å, ID 723 최대 1.750 Å
- ID 609에는 N-C-N 3-member smallest-ring basis와 RDKit이 보충한 implicit H 1개가 있어 구조적 신뢰도가 낮음
- ID 723은 implicit H와 소형 ring이 없어 topology는 상대적으로 더 자연스럽지만 고평면성·높은 logP·긴 aromatic bond 때문에 geometry optimization 후 판단 필요
- RDKit-valid만으로 안정성/OS를 판단할 수 없으며 xTB geometry optimization 및 sTDA 검증이 필요


## 2026-09-03 — 기존 5-class full-size best epoch 334 재생성

- 체크포인트 `epoch=334-step=246560.ckpt`, target f_osc 0.5, seed 46, all-guidance 1.0, max batch 32로 1,000개 생성
- sanitize 성공 7/1,000(0.7%), single-component/SMILES/SDF 성공 5/1,000(0.5%)
- 실패 분류: explicit valence 865, Kekulize 128, sanitize 후 multifragment 2
- 최종 5개는 모두 unique이며 train/val/test에 없는 novel molecule
- epoch 296 seed 45 결과의 sanitize 4개/SDF 2개보다 이번 seed에서는 증가했지만 표본이 작아 checkpoint 개선으로 단정할 수 없음
- SampleAnalyzer의 `frac_valid_mols=0.022`는 CSV의 strict sanitize+single-component 기준과 정의가 달라 최종 성공률로 사용하지 않음


## 2026-09-03 — epoch 334에서 10,000개 생성

- target f_osc 0.5, seed 47, all-guidance 1.0, max batch 32; 약 1시간 11분 소요
- sanitize 성공 69/10,000(0.69%), single-component/SMILES/SDF 성공 49/10,000(0.49%)
- 실패 분류: explicit valence 8,656, Kekulize 1,275, sanitize 후 multifragment 20
- 최종 49개는 모두 unique이며 train/val/test에 없는 novel molecule
- 직전 1,000개 seed 46의 strict 성공률 0.5%와 거의 동일하여 epoch 334 모델의 strict 생성 성공률은 약 0.5%로 재현됨


## 2026-09-03 — epoch 334 valid 49개 xTB/sTDA 평가

- `xtb_env`의 xTB 6.7.1과 별도 xtb4stda/sTDA 실행파일을 사용, 기존 `generated/xtb_stda_run`을 8-way 병렬 실행
- 49/49 xTB loose geometry optimization 수렴 및 정상 종료, 49/49 sTDA state 1~10 파싱 성공
- state 1~10 중 최대 OS: 평균 0.6150, 중앙값 0.5282, 범위 0.0825~2.3214
- 최대 OS가 target 0.5에서 0.05/0.1/0.2 이내: 8/13/18개
- 학습 target과 직접 대응하는 S1 OS: 평균 0.1655, 중앙값 0.0513, 범위 0~1.4358
- S1 OS가 target 0.5에서 0.05/0.1/0.2 이내: 3/3/4개
- S1 근접 상위: 01674=0.5205, 01949=0.4671, 07399=0.4582
- 기존 top-OS CSV는 S1이 아니라 state 1~10 최대값이므로 condition 반영률 평가에는 S1 값을 별도 사용해야 함
- top-1/top-3 CSV, 전체 grid, 7개 page grid, 분자별 PNG 49개 생성 완료

### state 1~10 OS top-5 기준 비교

- 사용자 판단 기준에 따라 S1 고정이 아니라 각 분자의 state 1~10을 OS 내림차순 정렬해 top-5로 비교
- rank별 OS 평균/중앙값: top1 0.6150/0.5282, top2 0.3330/0.2496, top3 0.2150/0.1571, top4 0.1567/0.1412, top5 0.1085/0.0717
- top1 state는 S10·S3 각 8개, S2·S1·S9 각 6개 등으로 분산; S1이 top1인 분자는 6/49뿐
- top-5 중 하나라도 target 0.5에서 0.05/0.1/0.2 이내인 분자: 13/24/29개
- target 최접근 후보: 01094(S4, rank3, 0.5007), 03135(S7, rank3, 0.4953), 01674(S1, rank2, 0.5205)
- top1 최대 후보: 05510(S9, 2.3214), 05092(S1, 1.4358), 03135(S2, 1.2991)
- long-format `stda_top5_os_state1-10.csv`와 molecule별 wide-format `stda_top5_os_comparison_wide.csv` 생성


## 2026-09-04 — 기존 5-class best epoch 483 생성 확인

- target f_osc 0.5, seed 48, all-guidance 1.0, max batch 32로 1,000개 생성
- sanitize 성공 7/1,000, single-component/SMILES/SDF 성공 6/1,000(0.6%)
- 실패 분류: explicit valence 731, Kekulize 262, sanitize 후 multifragment 1
- 최종 6개는 모두 unique 및 train/val/test에 없는 novel molecule
- epoch 334의 1,000개 strict 성공 5개와 큰 차이는 없지만 valence 실패는 865→731로 감소하고 Kekulize 실패는 128→262로 증가


## 2026-09-04 — 기존 5-class 모델 복원 기록 고정

- 학습은 epoch 533에서 early stopping으로 정상 종료; best checkpoint는 epoch 483
- `records/fullsize_5class_curriculum_recovery.md`에 source branch/commit, run config, checkpoint SHA-256, 종료 지표, 전체 generation 결과, xTB/sTDA 결과와 정확한 복원 명령 기록
- 실제 학습 출처는 `codex/condition-negative-curriculum` commit `f6eebe3529139f2f56b5e4b61a78b81e3e58024a`
- 현재 checkout된 `codex/kekule-bond-representation`의 4-class config와 기존 5-class checkpoint가 혼용되지 않도록 saved run config를 authoritative config로 명시
