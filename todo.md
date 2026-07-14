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

## 보류 (이번 라운드에서 손대지 않기로 결정)

- [ ] `utls/trainer.py`의 `WeightedIntentRQCFTrainer`, `ItemStableCFTrainer` 등 미사용으로
      보이는 클래스 — 논문에 실제로 쓰이는지 아직 불확실해서 보류
- [ ] tracked 노트북 3개(`prototype.ipynb`, `analyze/ablation_codebook_size.ipynb`,
      `analyze/find_best_score.ipynb`) — 공개 여부/정리 보류

## 남은 작업

- [ ] README.md의 `[TODO: ...]` 항목 채우기: 논문 제목/저자/abstract/arXiv 링크/BibTeX
- [ ] LICENSE 파일 추가 — 원본 PLD repo 라이선스 조건 확인 후 attribution 문구와 함께 결정
- [ ] 남은 `scripts/`·`config/` 중 논문 재현에 실제로 쓰이는 것만 추리기 (out-of-scope 데이터셋
      정리는 완료, 이제부턴 "쓰는 실험 vs 탐색용" 기준으로 정리). README의
      "Reproducing paper results" 섹션에 Table/Figure ↔ 스크립트 매핑 채우기
- [ ] `utls/trainer.py` 분리 검토 (2394줄, 트레이너 클래스 12개 — PRIDE 관련 코드만이라도
      별도 파일로 분리할지)
- [ ] `config/sweep_grid.yaml`이 깨져있음 — YAML이 아니라 `monitor.py` 소스 코드 내용이 그대로
      들어가 있어 파싱 불가. 사용 중인 파일인지 확인 후 복구/삭제 필요
