# GBL Hacker

Pokémon GO Battle League — Great League 팀 추천 엔진.
Taiman Party 메타 피드 + PvPoke 게임마스터 + set-state-aware 3v3 시뮬레이터로
`(expected_win_rate, worst_case_robustness, meta_coverage)` 3축 Pareto 프론티어를 산출한다.

---

## 셋업 (한 번만)

```bash
git clone https://github.com/blakkim/gbl-hacker.git
cd gbl-hacker
uv sync           # .python-version + uv.lock 기반으로 가상환경 + 의존성 설치
```

이후 `uv run gblh ...` 또는 활성화 후 `gblh ...` 로 실행.

---

## A. 현재 background 결과 재현 (동일 입력 → 동일 출력 보장)

리포에 박힌 fixture 스냅샷을 직접 가리켜 실행한다. 캐시·refresh 무관.

```bash
uv run gblh recommend \
  --path fixtures/snapshots/great_league__upper__2026-05-19.json \
  --top-k 5 \
  --lang ko \
  --candidates ranking \
  --pool-size 30 \
  --opponents ranking \
  --opponents-size 30 \
  --stochastic-samples 5 \
  --exclude 'サニーゴ(ガラル)'
```

**소요 시간**: ~110분 (M-series 기준). pool 30 × ordered 3-combo 23,352 후보 × 30 합성 상대 × N=5 샘플.

### 결정성 보장

- RNG seed = `md5(repr((team_a.species, team_b.species)))[:4]` (`src/gbl_hacker/score/expected_win_rate.py:462`)
  → 프로세스/머신 무관하게 동일 seed
- 데이터 입력 일체가 repo 안에 있음:
  - `fixtures/snapshots/great_league__upper__2026-05-19.json` — Taiman 스냅샷 (2026-05-19 05:40 UTC fetch)
  - `src/gbl_hacker/data/gamemaster.json` — PvPoke 미러
  - `src/gbl_hacker/data/rankings_gl.json` + `rankings_gl_leads.json` — PvPoke GL 랭킹
  - `src/gbl_hacker/data/cpm.json` / `type_chart.json` / `pokedex_localized.json` / `move_ja_to_pvpoke.json`

### 옵션 빠르게 보기

| 옵션 | 값 | 의미 |
|---|---|---|
| `--path` | fixture JSON | 캐시 대신 명시 스냅샷 사용 |
| `--candidates ranking` | (기본) | PvPoke top-N에서 ordered 3-combo enumerate (마스카나 같은 niche 픽 포함) |
| `--pool-size 30` |  | 풀 크기. 30 = 거의 전체 메타 (23K combo). 12=70초, 15=2.5분, 20=6분, 30=22~110분 |
| `--opponents ranking` | (기본) | PvPoke top 기반 합성 opponent 30팀 |
| `--stochastic-samples 5` | (기본) | 쉴드 결정 stochasticity 평균. WCR 0/100 collapse 방지 |
| `--exclude` | JP 종명 콤마구분 | 후보 풀에서 제외 (상대 풀은 안 빠짐) |
| `--win-mode` | `ko`(기본)/`resource` | 턴버짓 내 미KO 스톨 처리. `resource`=잔여자원(생존>쉴드>HP>에너지) 우위 팀에 승 |
| `--critique` | (플래그) | #1 팀에 red-team 비판: 팀 전체 공격 blind spot + 최악 시뮬 매치업 교차참조 |
| `--lang ko` | (기본) | 출력 종명 한국어 |

`--stochastic-samples 1` 로 바꾸면 결정론 모드(빠르지만 WCR 의미 없음, legacy).

---

## B. 신선한 데이터로 돌리기

오늘자 Taiman 메타로 새 스냅샷 받아서 추천:

```bash
# 1) 새 스냅샷 fetch + 캐시 저장 (~/.cache/gbl-hacker/snapshots/...)
uv run gblh refresh

# 2) 캐시 최신 스냅샷으로 추천 (--path 생략 시 가장 최근 캐시 사용)
uv run gblh recommend \
  --top-k 5 \
  --pool-size 30 \
  --opponents-size 30 \
  --exclude 'サニーゴ(ガラル)'
```

신선 데이터 = Taiman 업스트림 갱신 반영 → A의 결과와 다를 수 있음.

### 옵션 가볍게 (빠른 산책용)

```bash
uv run gblh refresh
uv run gblh recommend --pool-size 12 --opponents-size 20   # ~70초
```

---

## C. 캐시된 스냅샷 보기 (추천 없이)

```bash
uv run gblh show                    # 캐시 최신
uv run gblh show --path fixtures/snapshots/great_league__upper__2026-05-19.json
```

---

## D. 실전 ladder 결과 로깅

추천 팀으로 실제 GBL 셋 돌린 뒤 pre/post 레이팅을 JSONL 라인으로 기록:

```bash
uv run gblh report-rating --team-id <id> --pre 2845 --post 2861 --notes "altaria lead vs steel lead"
```

`seed.yaml`의 long-loop validation gate (rating_change_log ≥ 1) 충족 조건이기도 함.

---

## E. 다른 옵션·서브커맨드

- `uv run gblh refresh --help`
- `uv run gblh recommend --help`
- `uv run gblh show --help`
- `uv run gblh report-rating --help`
- `uv run gblh verify-reference --help` — 추천 결과를 독립 reference top-tier 리스트와 Jaccard 비교 (AC 5 gate)

---

## 결과 해석 노트

- **EWR**: 메타 사용률 가중 평균 승률
- **WCR**: 사용률 가중 10th-percentile 승률 (worst-case robustness, `--stochastic-samples > 1` 일 때만 의미 있음)
- **COV**: meta coverage — 메타 종 단위로 본 커버리지
- **Pareto size**: 세 축 모두 dominated 되지 않는 팀 개수. 1~2팀만 나오면 합성 상대 풀이 한쪽으로 쏠려 있을 가능성 (cf. WCR collapse caveat)

### 신뢰도 신호 (자동 출력)

추천 표 아래에 항상 따라 나온다 (숨길 플래그 없음):

- **TRUST — opponent-pool sensitivity**: top-K 팀을 *반대* 상대 풀(ranking↔meta) 양쪽에 채점한 EWR과 그 차이. gap ≥15%면 ⚠ — 그 순위는 상대 풀 선택에 의존한다는 뜻 (PvPoke 이론강팀엔 강하지만 실전 메타엔 약함, 또는 그 반대).
- **FRAGILE FRONTIER 경보**: `pareto_size ≤ 2`면 프론티어가 붕괴한 것 — 한 팀이 모든 축을 지배해서가 아니라 상대 풀이 쏠렸을 신호. top 픽을 잠정으로 취급하라는 경고.

### `--critique` red-team 리포트

- **offensive blind spots**: 팀 3마리의 공격 무브 타입을 모아, 그중 *어느 것도 super-effective가 아닌* 메타 종 (사용률순). per-mon 방어약점이 아니라 **팀 전체가 SE 레버리지 없는 적**을 잡는다 (예: 메더/파이어로/쏘콘 → Diggersby 노말/땅).
- **worst simulated matchups**: 메타 30팀 대비 시뮬 승률 최하위. blind-spot 종을 포함한 매치업엔 ⚠ — 가설(blind spot)을 시뮬(승률)로 확정한다.

### 빌드 정합성 (fact layer)

모든 빌드는 materialize 시 `gbl_hacker.coherence`로 검증된다 — 타이핑+무브가 단일 실제 gamemaster 폼으로 설명돼야 함. 어긋나면(과거 "メダ 키메라" 클래스) 기본 경고, `coherence="raise"`로 차단. 신선 스냅샷의 데이터 오염을 런타임에 잡는다.

데이터 신뢰성 caveat (Taiman은 report-density 가중 upper-bracket 피드)은 모든 `refresh`·`show` 출력에 구조적으로 박혀 나온다. 숨길 수 있는 플래그 없음.
