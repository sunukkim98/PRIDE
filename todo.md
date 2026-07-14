# TODO: 논문 공개용 repo 정비 (PRIDE 개명 + 리팩토링)

## 배경

논문에 이 repo 링크를 달기 위해 공개 준비를 한다. 두 가지 축:
1. 제안 방법 이름을 코드 전체에서 `MoEWarmupREQUIEM` → `PRIDE`로 변경
2. 공개 저장소로서 지저분한 부분 정리 (README, 개인 경로, 죽은 코드, 불필요 산출물)

완료된 이전 작업(`lambda_power` 중간 텐서 저장, `return_components` 플래그)은
`utls/trainer.py:1804` 기준으로 이미 구현되어 있어 이 문서에서 제거함.

---

## 1. 방법명 변경: `MoEWarmupREQUIEM` → `PRIDE`

`main.py`가 `--method` 문자열로 클래스/설정 함수를 동적으로 찾는 구조이므로
(`f"{method}CFTrainer"`, `f"get_{method}_config"`), 단순 문자열 치환이 아니라
아래 심볼들을 함께 바꿔야 동작한다.

- [ ] `utls/trainer.py:1303` — `class MoEWarmupREQUIEMCFTrainer` → `class PRIDECFTrainer`
- [ ] `utls/model_config.py:149` — `def get_MoEWarmupREQUIEM_config` → `def get_PRIDE_config`
- [ ] `meta_config.py:43` — `--method` 기본값 `'MoEWarmupREQUIEM'` → `'PRIDE'`
- [ ] `meta_config.py:76` — docstring/help 문구 내 `MoEWarmupREQUIEM` 언급 수정
- [ ] `utls/trainer.py` 내 로그 문자열 정리
  - `1980`: `f"[MoEREQUIEM] Epoch {epoch}: ..."` → `f"[PRIDE] Epoch {epoch}: ..."`
  - `2334`: `"[MoE Select] REQUIEM will continue on ..."` 문구도 함께 정리
- [ ] config yaml 전수 치환 — `method: ['MoEWarmupREQUIEM']` (150개 매치, 대부분 `config/**/*.yaml`)
  ```bash
  grep -rl "MoEWarmupREQUIEM" config/ | xargs sed -i "s/MoEWarmupREQUIEM/PRIDE/g"
  ```
- [ ] `scripts/*.slurm`, `scripts/*.sh` 내 `--method MoEWarmupREQUIEM` CLI 인자 치환
  (`run_single.sh`, `run_single.slurm`, `run_single_mf_mind*.slurm`, `run_mf_moerq_yelp_fair_select.*`, `run_stage2.py` 등)
- [ ] `--method` 값 변경 시 기존 `log/`, `checkpoints/` 경로가
      `.../MoEWarmupREQUIEM/...` → `.../PRIDE/...` 로 바뀌는 점 확인
      (경로가 `args.method`로 동적 생성됨 — `main.py:9`, `utls/trainer.py:88,1680,2166`)
- [ ] 이름 변경 후 `sbatch scripts/run_single_mf_mind.slurm` 한 번 더 돌려서
      회귀 없는지 확인 (base MF/MIND 결과: Recall@20 0.0801, NDCG@20 0.0530)
- [ ] 기반 클래스인 `REQUIEMCFTrainer`(`trainer.py:311`), `HybridREQUIEMCFTrainer`(`:454`)는
      PRIDE가 상속/참조하지 않는 별개 베이스라인이면 이름 유지 여부 확인 후 결정

---

## 2. 공개 저장소 정리

### 2-1. README 전면 재작성 (우선순위 최상)
- [ ] 현재 `README.md`가 **이 repo의 논문이 아니라 원본 PLD(WWW 2025, Kaike Zhang et al.) 논문 소개/인용**을 그대로 담고 있음.
      우리 논문 제목·저자·abstract·arXiv 링크·BibTeX로 전면 교체 필요.
- [ ] 원본 PLD 코드베이스를 기반으로 확장한 것이므로, 라이선스 허용 범위 내에서
      "Built on top of PLD (WWW 2025)" 식으로 원저자 attribution 유지할지 결정
- [ ] Usage 섹션을 PRIDE 기준 실행 예시로 교체 (`python main.py --model MF --dataset MIND --method PRIDE ...`)
- [ ] LICENSE 파일 없음 — 원본 PLD repo의 라이선스 조건 확인 후 추가

### 2-2. 개인 정보 / 하드코딩 경로 제거
- [ ] `scripts/run_single*.slurm` 등에 박힌 `~/miniconda3/envs/requiem/bin/python`,
      `cd ~/denoisevq` 절대경로 → `$CONDA_PREFIX`/상대경로/README 안내로 대체
- [ ] wandb entity/project(`forward-rec`, 개인 계정 `syi05003`)가 코드에 없는지 재확인
      (현재는 config 쪽엔 없고 실행 시 로그인 계정에서만 나옴 — 실제 하드코딩 없음, 최종 점검만)
- [ ] `git log` 커밋 작성자 중 `temp <config>` 더미 계정 이력 확인 — 그대로 공개해도 되는지 판단
      (내용 자체엔 문제 없어 보이나 review 필요)

### 2-3. `utls/trainer.py` 정리 (2394줄, 트레이너 클래스 12개)
- [ ] 논문에 실제로 비교/사용되는 baseline만 남기고, 탐색용으로 보이는
      `WeightedIntentRQCFTrainer`(`:625`), `ItemStableCFTrainer`(`:732`) 등
      미사용 클래스 삭제 여부 결정 (paper에 없는 ablation이면 제거)
- [ ] 죽은 주석 코드 제거 — 예: `:620` `# self._save_hist_png(...)` 같은 주석 처리된 블록
- [ ] 클래스가 많아 파일이 비대함 — `utls/trainer.py`를 `trainer/base.py`, `trainer/baselines.py`,
      `trainer/pride.py` 등으로 분리할지 검토 (최소한 PRIDE 관련 코드는 별도 파일로 분리 추천)

### 2-4. 저장소에 쌓인 산출물/불필요 파일 정리
- [ ] `.pyc`, `__pycache__/` 가 다수 커밋되어 있음 (`git ls-files | grep __pycache__`) —
      `git rm -r --cached **/__pycache__` 후 `.gitignore`에 `__pycache__/`, `*.pyc` 추가
      (현재 `.gitignore`엔 빠져있음)
- [ ] 탐색용 노트북 3개가 tracked 상태: `prototype.ipynb`, `analyze/ablation_codebook_size.ipynb`,
      `analyze/find_best_score.ipynb` — 공개 repo에 남길지, `analyze/` 정식 스크립트로 정리할지 결정
      (`.gitignore`에 `analyze/`가 있는데도 이미 커밋된 파일이라 계속 tracked됨 — 강제 정리 필요)
- [ ] `.gitignore`의 `*.txt` 규칙이 `requirements.txt`까지 잡을 수 있는 범위인데
      현재는 이미 tracked라 무시되고 있음 — 규칙을 `*.txt`보다 좁혀서 의도치 않은 텍스트 산출물만 막도록 수정

### 2-5. `scripts/`(67개) · `config/`(196개) 정리
- [ ] 실험 스크립트가 매우 많음 — 논문 결과 재현에 필요한 것만 추려서
      `scripts/`, `config/`를 `paper/`(재현용) vs `scripts/experimental/`(탐색용)로 구분하거나,
      불필요한 것은 삭제
- [ ] `README.md`에 "논문 Table X 재현 = scripts/xxx.slurm" 식 매핑 안내 추가

---

## 진행 순서 제안

1. 방법명 변경 (섹션 1) — 실행 가능 여부부터 확정
2. README/LICENSE 재작성 (섹션 2-1) — 외부 공개 전 필수
3. 개인 경로 제거 (2-2), pycache/notebook 정리 (2-4) — 기계적 정리
4. trainer.py 리팩토링 (2-3), scripts/config 정리 (2-5) — 시간 여유 봐서 진행
