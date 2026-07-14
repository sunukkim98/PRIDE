# TODO: 논문 공개용 repo 정비 (PRIDE 개명 + 리팩토링)

## 완료

- [x] git 상태 정리: 마지막 커밋(2025-11-13) 이후 8개월치 미커밋 작업(trainer.py 등 실코드 변경 +
      scripts/config 169개 신규 파일)을 하나의 스냅샷 커밋(`54a0437`)으로 반영
- [x] 방법명 변경: `MoEWarmupREQUIEM` → `PRIDE`
  - `utls/trainer.py`: `MoEWarmupREQUIEMCFTrainer` → `PRIDECFTrainer`, 로그 문자열(`[MoEREQUIEM]` 등) 정리
  - `utls/model_config.py`: `get_MoEWarmupREQUIEM_config` → `get_PRIDE_config`
  - `meta_config.py`: `--method` 기본값 및 help 문구
  - config yaml 150곳, scripts 내 `--method` 인자 일괄 치환
  - 로컬 2-epoch 스모크 런 + slurm 100-epoch 풀 런(job 96317)으로 회귀 없음 확인
    (Recall@20 0.0801 / NDCG@20 0.0530, 기존과 동일)
- [x] `scripts/run_single*.slurm` 개인 경로 제거
  - `cd ~/denoisevq` → `cd "$SLURM_SUBMIT_DIR"` (dirname "$0"은 sbatch가 스크립트를 spool로
    복사해서 실행하기 때문에 안 됨 — 실제로 한 번 깨져서 확인함)
  - `~/miniconda3/envs/requiem/bin/python` → `conda activate requiem` 후 `python`
- [x] `.gitignore` 보강 (`__pycache__/`, `*.pyc`, `.codex`, `tea_debug.log` 추가, 애매했던
      `*.txt` 규칙 제거) + 기존 커밋된 `__pycache__`/`.pyc` untrack
- [x] README.md 스켈레톤 재작성 — 기존 PLD 논문 소개를 PRIDE 기준 구조(Environment/Usage/
      Repository Structure)로 교체. 논문 제목/저자/abstract/BibTeX/원저자 attribution 문구는
      `[TODO: ...]` placeholder로 남김 (실제 논문 정보 확정되면 채우기)
- [x] 실험 대상 데이터셋을 **Amazon-Book, Yelp, MIND** 3개로 확정 — README Datasets 섹션 반영
      (기존에 나열했던 Baby/Gowalla/Office/Software/Toys_and_Games/Twitch 제외)
- [x] 범위 밖 데이터셋 config 41개(전부 `*_toys.yaml`, Toys_and_Games 단독 sweep)를 삭제하지 않고
      `.gitignore`(`*_toys.yaml`)로만 제외 + `git rm --cached`로 untrack (로컬 파일은 유지).
      scripts는 전부 제외 대상 아님으로 판명 — `run_neumf.slurm` 등은 Toys_and_Games/Gowalla를
      포함해 여러 데이터셋을 loop 돌리는 구조라 Amazon-Book/Yelp/MIND도 같이 커버함
      (`sweep_neumf_origin.yaml`도 매 dataset마다 동적으로 덮어쓰는 템플릿이라 유지)
- [x] README에 논문 제목 반영: "Preference-structure and Representation-stability based
      Interaction Denoising"
- [x] `REQUIEM`, `HybridREQUIEM` 완전 제거 — PRIDE(구 MoEWarmupREQUIEM)의 옛 프로토타입이라
      더 이상 안 씀. `REQUIEMCFTrainer`/`HybridREQUIEMCFTrainer`/`get_REQUIEM_config`/
      `get_HybridREQUIEM_config` 삭제, README `--method` 표와 attribution note에서도 제거.
      기존 config/scripts 중 이 두 method를 쓰는 곳 없어서 안전하게 삭제 확인함
- [x] LICENSE 처리 방향 결정 — PLD 원본 repo(https://github.com/Kaike-Zhang/PLD)에
      LICENSE 파일이 없음(GitHub API: `license: null`, 즉 all-rights-reserved 상태)을 확인.
      우리 repo에도 별도 LICENSE 파일을 추가하지 않고, README에 attribution 문구만 남기기로
      결정 (원본 repo 링크 + 감사 인사로 정리 완료)
- [x] README를 연구실 이전 논문(AlphaFree) repo 형식에 맞춰 전면 재작성
      - Datasets 표에 실제 통계 반영 (CFDataset 직접 로드해서 측정: Amazon-Book 52,643 유저/
        91,599 아이템/2,704,860 상호작용, Yelp 31,668/38,048/1,561,406, MIND 38,441/38,000/1,210,953)
      - Validated hyperparameters 표에 `config/noise_mf_{dataset}.yaml` 기준 실측 값 반영
        (3개 데이터셋 모두 확인됨 — lr/weight_decay/begin_adv/ema/num_codebook/energy_r/
        energy_lambda/beta/drop_rate/num_gradual)
      - AlphaFree엔 없는 Preprocessing/Inference phase 구분, C++ 컴파일, download.sh, Demo
        섹션은 우리 repo에 해당 사항이 없어서 제외
      - Performance 비교 표(baseline 대비 PRIDE 결과)는 `analyze/csv/`에 후보 숫자가 있긴
        하지만 최종 확정치인지 불확실해서 채우지 않고 TODO로 남김 — 확정되면 채우기
      - 기존에 있던 "Repository Structure" 커스텀 섹션은 AlphaFree 형식에 없어서 제거함
        (필요하면 복원 가능)

## 보류 (이번 라운드에서 손대지 않기로 결정)

- [ ] `utls/trainer.py`의 `WeightedIntentRQCFTrainer`, `ItemStableCFTrainer` 등 미사용으로
      보이는 클래스 — 논문에 실제로 쓰이는지 아직 불확실해서 보류
- [ ] tracked 노트북 3개(`prototype.ipynb`, `analyze/ablation_codebook_size.ipynb`,
      `analyze/find_best_score.ipynb`) — 공개 여부/정리 보류

- [x] 실제 논문 PDF를 받아서 README를 진짜 데이터로 완성
      - 저자(Sunuk Kim, Minseo Jeon 공동1저자, Daewon Gwak, Gyuwon Je, Jinhong Jung 교신), abstract,
        키워드, 데이터셋 설명(Yelp2018/MIND/Amazon-Book 출처: PLD repo / LightGCN++ repo) 반영
      - Table 2(데이터 통계)가 내가 직접 측정한 값과 정확히 일치함을 확인 (교차검증 완료)
      - Table 3(MF/LightGCN 성능 비교, baseline 6개 대비 PRIDE), Table 4/5(ablation),
        Appendix A(하이퍼파라미터 탐색 범위) 논문에서 그대로 전사해서 채움
      - 논문 게재 상태가 "Preprint submitted to Elsevier"로, 아직 accept된 저널/DOI 없음 —
        BibTeX는 확정 전까지 TODO로 유지

## 발견한 문제 — 확인 필요

- [x] ~~하이퍼파라미터 불일치~~ → 논문 Figure 3 선택값(MIND MF: num_codebook=1024/ema=0.99/
      energy_lambda=0.9/begin_adv=10)이 repo config(512/0.75/0.5/15)와 다른 걸 발견해서 물어봤으나,
      사용자가 "무시하고 README에서 해당 경고 문구 삭제"로 결정함. repo config 자체는 그대로 두고
      README엔 더 이상 이 불일치를 언급하지 않음 (Figure 3 선택값 표 자체는 논문 그대로 유지)
- [x] ~~ablation 이름 매핑 불확실~~ → README에서 Ablation study 섹션 자체를 삭제하기로 결정해서
      해소됨 (사용자 요청). 코드상 `wo_requiem` vs 논문의 "w/o Warm-up Stage" 매핑 문제는 README엔
      더 이상 안 나오지만, 실제 재현 스크립트 작성 시엔 여전히 유효한 질문으로 남아있음

## 남은 작업

- [ ] BibTeX — 저널 accept 후 갱신
- [ ] 데이터 다운로드/전처리 커맨드 구체화 (PLD repo·LightGCN++ repo에서 어떤 파일을 받아
      `data/<Dataset>/data.json`으로 만드는지)
- [ ] 남은 `scripts/`·`config/` 중 논문 재현에 실제로 쓰이는 것만 추리기 (out-of-scope 데이터셋
      정리는 완료, 이제부턴 "쓰는 실험 vs 탐색용" 기준으로 정리)
- [ ] `utls/trainer.py` 분리 검토 (2394줄, 트레이너 클래스 12개 — PRIDE 관련 코드만이라도
      별도 파일로 분리할지)
- [ ] `config/sweep_grid.yaml`이 깨져있음 — YAML이 아니라 `monitor.py` 소스 코드 내용이 그대로
      들어가 있어 파싱 불가. 사용 중인 파일인지 확인 후 복구/삭제 필요
