#!/usr/bin/env python3
"""
LLM token throughput maximizer — OpenAI-compatible Chat Completions API.

Objectif :
  Maximiser le débit réel de tokens sur une API /v1/chat/completions.

Principes :
  1. Warmup court.
  2. Discovery par niveaux de concurrence : 1,2,4,8,...
  3. Mesure par plateau temporel, pas par batch unique.
  4. Sélection du meilleur niveau selon output_tokens/sec prioritaire.
  5. Exploitation avec worker pool continu :
       - C requêtes en vol.
       - Dès qu'une requête finit, une autre est relancée.
  6. Probes occasionnels C+STEP et C-STEP.
  7. Export JSON + CSV.

Pourquoi cette version :
  - Pas d'asyncio.run() imbriqué.
  - Compteurs réellement incrémentés.
  - Pas de trous entre les batches.
  - Séparation prompt_tokens / completion_tokens.
  - Sélection du meilleur C par médiane output_tokens/sec.
  - Deadline absolu, donc durée totale respectée.

Variables .env utiles :
  API_URL=http://localhost:8000/v1/chat/completions
  API_KEY=
  MODEL=your-model-name

  STRESS_DURATION=300
  WARMUP_SEC=10
  DISCOVERY_PLATEAU_SEC=20
  MAX_CONCURRENCY=64
  REQUEST_TIMEOUT=180

  MAX_TOKENS=768
  TEMPERATURE=0.7
  TOP_P=0.95

  EXPLOIT_PROBE_INTERVAL=60
  STEP_SIZE=2
  MIN_SUCCESS_RATE=0.95

  OUTPUT_JSON=stress_result.json
  OUTPUT_CSV=stress_result.csv
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000/v1/chat/completions")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("MODEL", "")

STRESS_DURATION = int(os.getenv("STRESS_DURATION", "300"))
WARMUP_SEC = int(os.getenv("WARMUP_SEC", "10"))
DISCOVERY_PLATEAU_SEC = int(os.getenv("DISCOVERY_PLATEAU_SEC", "20"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "64"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "768"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

EXPLOIT_PROBE_INTERVAL = int(os.getenv("EXPLOIT_PROBE_INTERVAL", "60"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "2"))
MIN_SUCCESS_RATE = float(os.getenv("MIN_SUCCESS_RATE", "0.95"))

OUTPUT_JSON = os.getenv("OUTPUT_JSON", "stress_result.json")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "stress_result.csv")

# Pour éviter de tuer brutalement un serveur fragile.
STOP_ON_TOTAL_FAILURE = os.getenv("STOP_ON_TOTAL_FAILURE", "true").lower() in {"1", "true", "yes", "y"}

# Si le serveur ne renvoie pas usage, on peut estimer grossièrement ou compter 0.
# Pour un benchmark sérieux, il vaut mieux que le serveur renvoie usage.prompt_tokens/completion_tokens.
ALLOW_ZERO_USAGE = os.getenv("ALLOW_ZERO_USAGE", "true").lower() in {"1", "true", "yes", "y"}


# ============================================================
# Prompts contrôlés
# ============================================================

LONG_OUTPUT_PROMPTS = [
    """Écris une analyse structurée et détaillée sur l'optimisation des systèmes distribués.
Développe plusieurs sections, donne des exemples concrets, explique les compromis, et continue jusqu'à produire une réponse longue et complète.""",

    """Rédige un guide technique complet sur le déploiement d'un modèle de langage en production.
Inclure architecture, monitoring, sécurité, performance, cache, files d'attente, haute disponibilité et limites opérationnelles.""",

    """Explique en détail les différences entre prefill, decode, KV cache, batching continu et speculative decoding dans les serveurs LLM.
Ajoute des exemples, des cas limites, et des recommandations pratiques.""",

    """Écris une étude complète sur les stratégies de réduction de latence pour une API LLM.
Traite la concurrence, le streaming, la taille du contexte, la taille de sortie, le scheduling et la saturation GPU.""",

    """Rédige un rapport approfondi sur la conception d'un benchmark de performance pour une API compatible OpenAI.
Décris les métriques, erreurs fréquentes, biais de mesure, choix des prompts, statistiques, et interprétation des résultats.""",
]


SHORT_OUTPUT_PROMPTS = [
    "Résume en 5 points les avantages du batching continu pour les LLM.",
    "Explique brièvement ce qu'est le KV cache.",
    "Donne trois causes possibles d'une latence P95 élevée sur une API LLM.",
    "Liste les métriques essentielles pour benchmarker un serveur LLM.",
    "Explique la différence entre throughput et latence.",
]


MIXED_PROMPTS = LONG_OUTPUT_PROMPTS + SHORT_OUTPUT_PROMPTS


def pick_prompt(mode: str, idx: int) -> str:
    """
    mode:
      - long_output : maximise completion tokens/sec
      - short_output : simule usage court
      - mixed : usage plus réaliste
    """
    if mode == "long_output":
        arr = LONG_OUTPUT_PROMPTS
    elif mode == "short_output":
        arr = SHORT_OUTPUT_PROMPTS
    else:
        arr = MIXED_PROMPTS

    # Décalage deterministic + petite diversité
    base = arr[idx % len(arr)]
    return f"{base}\n\nIdentifiant de variation: {idx}. Ne te limite pas à une réponse trop courte."


# ============================================================
# Data structures
# ============================================================

@dataclass
class RequestResult:
    ok: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    status: int
    error: str
    phase: str
    concurrency: int
    started_at: float
    ended_at: float


@dataclass
class PlateauSummary:
    phase: str
    concurrency: int
    duration_sec: float
    requests_total: int
    requests_ok: int
    requests_fail: int
    success_rate: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_per_sec: float
    completion_tokens_per_sec: float
    total_tokens_per_sec: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    errors: dict[str, int]


@dataclass
class RunSummary:
    duration_sec: float
    total_requests: int
    success: int
    fail: int
    success_rate: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_prompt_tokens_per_sec: float
    avg_completion_tokens_per_sec: float
    avg_total_tokens_per_sec: float
    optimal_concurrency: int
    best_completion_tokens_per_sec: float
    best_total_tokens_per_sec: float


# ============================================================
# Stats helpers
# ============================================================

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize_results(
    *,
    phase: str,
    concurrency: int,
    duration_sec: float,
    results: list[RequestResult],
) -> PlateauSummary:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    lats = [r.latency_ms for r in results]

    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    total_tokens = sum(r.total_tokens for r in ok)

    errors: dict[str, int] = {}
    for r in fail:
        key = r.error or f"HTTP{r.status}" or "unknown"
        errors[key] = errors.get(key, 0) + 1

    dur = max(duration_sec, 0.001)

    return PlateauSummary(
        phase=phase,
        concurrency=concurrency,
        duration_sec=round(duration_sec, 3),
        requests_total=len(results),
        requests_ok=len(ok),
        requests_fail=len(fail),
        success_rate=(len(ok) / len(results)) if results else 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_per_sec=prompt_tokens / dur,
        completion_tokens_per_sec=completion_tokens / dur,
        total_tokens_per_sec=total_tokens / dur,
        latency_avg_ms=(sum(lats) / len(lats)) if lats else 0.0,
        latency_p50_ms=percentile(lats, 0.50),
        latency_p90_ms=percentile(lats, 0.90),
        latency_p95_ms=percentile(lats, 0.95),
        latency_p99_ms=percentile(lats, 0.99),
        errors=errors,
    )


def print_plateau(s: PlateauSummary) -> None:
    print(
        f"  C={s.concurrency:<4d} | "
        f"OK:{s.requests_ok:>4d}/{s.requests_total:<4d} "
        f"SR:{s.success_rate*100:>5.1f}% | "
        f"out:{s.completion_tokens_per_sec:>8.1f} tok/s | "
        f"total:{s.total_tokens_per_sec:>8.1f} tok/s | "
        f"P50:{s.latency_p50_ms/1000:>5.1f}s "
        f"P95:{s.latency_p95_ms/1000:>5.1f}s"
    )


def progress_bar(pct: float, width: int = 30) -> str:
    pct = max(0.0, min(1.0, pct))
    n = int(pct * width)
    return "█" * n + "░" * (width - n)


# ============================================================
# HTTP request
# ============================================================

async def send_request(
    session: aiohttp.ClientSession,
    *,
    prompt_idx: int,
    phase: str,
    concurrency: int,
    prompt_mode: str,
) -> RequestResult:
    prompt = pick_prompt(prompt_mode, prompt_idx)
    started = time.monotonic()

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    try:
        async with session.post(API_URL, json=payload) as resp:
            text = await resp.text()
            ended = time.monotonic()
            latency_ms = (ended - started) * 1000

            if resp.status != 200:
                err = ""
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        err_obj = data.get("error", "")
                        if isinstance(err_obj, dict):
                            err = err_obj.get("message", "") or err_obj.get("type", "")
                        else:
                            err = str(err_obj)
                except Exception:
                    err = text[:200] if text else ""

                return RequestResult(
                    ok=False,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    status=resp.status,
                    error=f"HTTP{resp.status}:{err[:120]}",
                    phase=phase,
                    concurrency=concurrency,
                    started_at=started,
                    ended_at=ended,
                )

            try:
                data = json.loads(text)
            except Exception:
                return RequestResult(
                    ok=False,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    status=resp.status,
                    error="invalid_json",
                    phase=phase,
                    concurrency=concurrency,
                    started_at=started,
                    ended_at=ended,
                )

            if not data.get("choices"):
                return RequestResult(
                    ok=False,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    status=resp.status,
                    error="no_choices",
                    phase=phase,
                    concurrency=concurrency,
                    started_at=started,
                    ended_at=ended,
                )

            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

            if not ALLOW_ZERO_USAGE and total_tokens <= 0:
                return RequestResult(
                    ok=False,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    status=resp.status,
                    error="missing_usage",
                    phase=phase,
                    concurrency=concurrency,
                    started_at=started,
                    ended_at=ended,
                )

            return RequestResult(
                ok=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                status=resp.status,
                error="",
                phase=phase,
                concurrency=concurrency,
                started_at=started,
                ended_at=ended,
            )

    except asyncio.TimeoutError:
        ended = time.monotonic()
        return RequestResult(
            ok=False,
            latency_ms=(ended - started) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            status=0,
            error="timeout",
            phase=phase,
            concurrency=concurrency,
            started_at=started,
            ended_at=ended,
        )
    except aiohttp.ClientError as e:
        ended = time.monotonic()
        return RequestResult(
            ok=False,
            latency_ms=(ended - started) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            status=0,
            error=f"aiohttp:{type(e).__name__}",
            phase=phase,
            concurrency=concurrency,
            started_at=started,
            ended_at=ended,
        )
    except Exception as e:
        ended = time.monotonic()
        return RequestResult(
            ok=False,
            latency_ms=(ended - started) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            status=0,
            error=f"err:{type(e).__name__}",
            phase=phase,
            concurrency=concurrency,
            started_at=started,
            ended_at=ended,
        )


# ============================================================
# Continuous worker pool
# ============================================================

async def run_continuous_plateau(
    session: aiohttp.ClientSession,
    *,
    phase: str,
    concurrency: int,
    duration_sec: float,
    prompt_mode: str,
    prompt_counter_start: int,
    global_deadline: Optional[float] = None,
    live: bool = False,
) -> tuple[PlateauSummary, list[RequestResult], int]:
    """
    Maintient exactement `concurrency` requêtes en vol pendant `duration_sec`.

    Dès qu'une requête termine, on en relance une autre jusqu'à la deadline.
    C'est beaucoup plus efficace qu'un batch qui attend toujours le plus lent.
    """
    start = time.monotonic()
    local_deadline = start + duration_sec
    deadline = min(local_deadline, global_deadline) if global_deadline else local_deadline

    results: list[RequestResult] = []
    prompt_idx = prompt_counter_start
    in_flight: set[asyncio.Task[RequestResult]] = set()

    def create_one() -> asyncio.Task[RequestResult]:
        nonlocal prompt_idx
        task = asyncio.create_task(
            send_request(
                session,
                prompt_idx=prompt_idx,
                phase=phase,
                concurrency=concurrency,
                prompt_mode=prompt_mode,
            )
        )
        prompt_idx += 1
        return task

    for _ in range(max(concurrency, 1)):
        in_flight.add(create_one())

    last_print = 0.0

    while in_flight:
        now = time.monotonic()
        if now >= deadline:
            break

        timeout = max(0.05, min(1.0, deadline - now))
        done, pending = await asyncio.wait(
            in_flight,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            if live and now - last_print >= 0.5:
                last_print = now
                elapsed = now - start
                pct = elapsed / max(duration_sec, 0.001)
                ok_so_far = sum(1 for r in results if r.ok)
                comp_tok = sum(r.completion_tokens for r in results if r.ok)
                tps = comp_tok / max(elapsed, 0.001)
                print(
                    f"\r  [{progress_bar(pct)}] "
                    f"{pct*100:>5.1f}% | C:{concurrency:<4d} | "
                    f"req:{len(results):>5d} ok:{ok_so_far:>5d} | "
                    f"out:{tps:>8.1f} tok/s",
                    end="",
                    flush=True,
                )
            continue

        for task in done:
            in_flight.discard(task)
            try:
                results.append(task.result())
            except Exception as e:
                end = time.monotonic()
                results.append(
                    RequestResult(
                        ok=False,
                        latency_ms=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        status=0,
                        error=f"task_exception:{type(e).__name__}",
                        phase=phase,
                        concurrency=concurrency,
                        started_at=end,
                        ended_at=end,
                    )
                )

            if time.monotonic() < deadline:
                in_flight.add(create_one())

        if live and now - last_print >= 0.5:
            last_print = now
            elapsed = now - start
            pct = elapsed / max(duration_sec, 0.001)
            ok_so_far = sum(1 for r in results if r.ok)
            fail_so_far = len(results) - ok_so_far
            comp_tok = sum(r.completion_tokens for r in results if r.ok)
            total_tok = sum(r.total_tokens for r in results if r.ok)
            out_tps = comp_tok / max(elapsed, 0.001)
            total_tps = total_tok / max(elapsed, 0.001)
            lats = [r.latency_ms for r in results]
            p95 = percentile(lats, 0.95) / 1000 if lats else 0

            print(
                f"\r  [{progress_bar(pct)}] "
                f"{pct*100:>5.1f}% | C:{concurrency:<4d} | "
                f"req:{len(results):>5d} ok:{ok_so_far:>5d} fail:{fail_so_far:<4d} | "
                f"out:{out_tps:>8.1f} tok/s total:{total_tps:>8.1f} tok/s | "
                f"P95:{p95:>5.1f}s",
                end="",
                flush=True,
            )

    # On annule les requêtes encore en vol à la fin du plateau.
    # Pour un test serveur, c'est acceptable : la mesure est bornée temporellement.
    for task in in_flight:
        task.cancel()

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)

    end = time.monotonic()
    duration = max(end - start, 0.001)

    if live:
        print()

    summary = summarize_results(
        phase=phase,
        concurrency=concurrency,
        duration_sec=duration,
        results=results,
    )

    return summary, results, prompt_idx


# ============================================================
# Discovery
# ============================================================

async def warmup(
    session: aiohttp.ClientSession,
    *,
    prompt_mode: str,
    prompt_counter: int,
) -> int:
    if WARMUP_SEC <= 0:
        return prompt_counter

    print("\n" + "=" * 72)
    print(f"🔥 WARMUP — {WARMUP_SEC}s à C=1")
    print("=" * 72)

    s, _, prompt_counter = await run_continuous_plateau(
        session,
        phase="warmup",
        concurrency=1,
        duration_sec=WARMUP_SEC,
        prompt_mode=prompt_mode,
        prompt_counter_start=prompt_counter,
        live=True,
    )
    print_plateau(s)
    return prompt_counter


async def discovery(
    session: aiohttp.ClientSession,
    *,
    prompt_mode: str,
    prompt_counter: int,
    global_deadline: float,
) -> tuple[int, list[PlateauSummary], list[RequestResult], int]:
    """
    Discovery par paliers exponentiels.
    Stop :
      - max concurrency atteint
      - success rate trop bas
      - débit completion_tokens/sec baisse franchement après un bon point
      - deadline proche
    """
    print("\n" + "=" * 72)
    print("🔍 DISCOVERY — paliers de concurrence")
    print("=" * 72)

    summaries: list[PlateauSummary] = []
    all_results: list[RequestResult] = []

    conc = 1
    previous_good: Optional[PlateauSummary] = None
    best_good: Optional[PlateauSummary] = None

    while conc <= MAX_CONCURRENCY:
        remaining = global_deadline - time.monotonic()
        if remaining < max(5, DISCOVERY_PLATEAU_SEC * 0.5):
            print("  ⏹️  Discovery stoppée : temps restant insuffisant.")
            break

        plateau_duration = min(DISCOVERY_PLATEAU_SEC, max(5, remaining * 0.35))

        s, results, prompt_counter = await run_continuous_plateau(
            session,
            phase="discovery",
            concurrency=conc,
            duration_sec=plateau_duration,
            prompt_mode=prompt_mode,
            prompt_counter_start=prompt_counter,
            global_deadline=global_deadline,
            live=False,
        )

        summaries.append(s)
        all_results.extend(results)
        print_plateau(s)

        if s.requests_total == 0:
            print("  ❌ Aucun résultat collecté à ce niveau.")
            break

        if s.success_rate < MIN_SUCCESS_RATE:
            print(
                f"  ❌ C={conc} rejeté : success rate "
                f"{s.success_rate*100:.1f}% < {MIN_SUCCESS_RATE*100:.1f}%."
            )
            if STOP_ON_TOTAL_FAILURE and s.requests_ok == 0:
                break
            # Si C échoue, inutile de monter plus haut.
            break

        if best_good is None or s.completion_tokens_per_sec > best_good.completion_tokens_per_sec:
            best_good = s

        # Stop intelligent :
        # Si on a déjà un bon point et que le débit baisse de >15% au niveau suivant,
        # monter encore n'est souvent pas utile.
        if previous_good is not None:
            if s.completion_tokens_per_sec < previous_good.completion_tokens_per_sec * 0.85:
                print(
                    f"  ⚠️  Débit en baisse nette à C={conc}; "
                    f"meilleur provisoire C={best_good.concurrency if best_good else previous_good.concurrency}."
                )
                break

        previous_good = s
        conc *= 2

    if not best_good:
        # Fallback minimal
        best_c = 1
    else:
        best_c = best_good.concurrency

    print(f"\n✅ Discovery terminée → C optimal provisoire = {best_c}")
    return best_c, summaries, all_results, prompt_counter


# ============================================================
# Exploitation with probes
# ============================================================

async def run_probe(
    session: aiohttp.ClientSession,
    *,
    base_summary: PlateauSummary,
    candidate_conc: int,
    prompt_mode: str,
    prompt_counter: int,
    global_deadline: float,
) -> tuple[PlateauSummary, list[RequestResult], int]:
    probe_duration = min(10.0, max(4.0, DISCOVERY_PLATEAU_SEC * 0.4))
    remaining = global_deadline - time.monotonic()
    probe_duration = min(probe_duration, max(1.0, remaining))

    print(f"\n  🔎 Probe C={candidate_conc} pendant {probe_duration:.1f}s")

    s, results, prompt_counter = await run_continuous_plateau(
        session,
        phase="probe",
        concurrency=candidate_conc,
        duration_sec=probe_duration,
        prompt_mode=prompt_mode,
        prompt_counter_start=prompt_counter,
        global_deadline=global_deadline,
        live=False,
    )

    print_plateau(s)
    return s, results, prompt_counter


async def exploitation(
    session: aiohttp.ClientSession,
    *,
    initial_concurrency: int,
    prompt_mode: str,
    prompt_counter: int,
    global_deadline: float,
) -> tuple[int, list[PlateauSummary], list[RequestResult], int]:
    print("\n" + "=" * 72)
    print(f"🚀 EXPLOITATION — worker pool continu, départ C={initial_concurrency}")
    print("=" * 72)

    current_c = max(1, min(initial_concurrency, MAX_CONCURRENCY))
    summaries: list[PlateauSummary] = []
    all_results: list[RequestResult] = []

    last_probe = time.monotonic()
    best_summary: Optional[PlateauSummary] = None

    while True:
        remaining = global_deadline - time.monotonic()
        if remaining <= 2:
            break

        # Plateau d'exploitation entre deux probes.
        plateau_duration = min(EXPLOIT_PROBE_INTERVAL, remaining)

        s, results, prompt_counter = await run_continuous_plateau(
            session,
            phase="exploitation",
            concurrency=current_c,
            duration_sec=plateau_duration,
            prompt_mode=prompt_mode,
            prompt_counter_start=prompt_counter,
            global_deadline=global_deadline,
            live=True,
        )

        summaries.append(s)
        all_results.extend(results)
        print_plateau(s)

        if best_summary is None or s.completion_tokens_per_sec > best_summary.completion_tokens_per_sec:
            best_summary = s

        remaining = global_deadline - time.monotonic()
        if remaining <= 8:
            break

        # Probe léger autour du C courant.
        # On essaie d'abord de monter, car objectif = maximisation de tokens.
        candidates: list[int] = []

        up = min(MAX_CONCURRENCY, current_c + STEP_SIZE)
        down = max(1, current_c - STEP_SIZE)

        if up != current_c:
            candidates.append(up)
        if down != current_c and remaining > 20:
            candidates.append(down)

        accepted = False
        for cand in candidates:
            if global_deadline - time.monotonic() <= 8:
                break

            probe_s, probe_results, prompt_counter = await run_probe(
                session,
                base_summary=s,
                candidate_conc=cand,
                prompt_mode=prompt_mode,
                prompt_counter=prompt_counter,
                global_deadline=global_deadline,
            )

            summaries.append(probe_s)
            all_results.extend(probe_results)

            if probe_s.success_rate < MIN_SUCCESS_RATE:
                continue

            # On exige une amélioration nette pour éviter l'oscillation.
            if probe_s.completion_tokens_per_sec > s.completion_tokens_per_sec * 1.05:
                print(
                    f"  📈 Changement accepté : C={current_c} → C={cand} "
                    f"({s.completion_tokens_per_sec:.1f} → "
                    f"{probe_s.completion_tokens_per_sec:.1f} out tok/s)"
                )
                current_c = cand
                accepted = True
                break

        if not accepted:
            print(f"  ✅ C maintenu à {current_c}")

        last_probe = time.monotonic()

    print(f"\n✅ Exploitation terminée → C final = {current_c}")
    return current_c, summaries, all_results, prompt_counter


# ============================================================
# Reporting
# ============================================================

def group_by_concurrency(summaries: list[PlateauSummary]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[PlateauSummary]] = {}

    for s in summaries:
        if s.phase == "warmup":
            continue
        grouped.setdefault(s.concurrency, []).append(s)

    out: dict[int, dict[str, Any]] = {}

    for c, arr in grouped.items():
        completion_tps = [s.completion_tokens_per_sec for s in arr if s.success_rate >= MIN_SUCCESS_RATE]
        total_tps = [s.total_tokens_per_sec for s in arr if s.success_rate >= MIN_SUCCESS_RATE]
        success_rates = [s.success_rate for s in arr]

        out[c] = {
            "measurements": len(arr),
            "median_completion_tokens_per_sec": median(completion_tps),
            "max_completion_tokens_per_sec": max(completion_tps) if completion_tps else 0.0,
            "median_total_tokens_per_sec": median(total_tps),
            "max_total_tokens_per_sec": max(total_tps) if total_tps else 0.0,
            "median_success_rate": median(success_rates),
            "phases": sorted(set(s.phase for s in arr)),
        }

    return out


def choose_best_concurrency(summaries: list[PlateauSummary]) -> tuple[int, float, float]:
    grouped = group_by_concurrency(summaries)

    candidates = {
        c: v for c, v in grouped.items()
        if v["median_success_rate"] >= MIN_SUCCESS_RATE
        and v["median_completion_tokens_per_sec"] > 0
    }

    if not candidates:
        return 1, 0.0, 0.0

    best_c, best_v = max(
        candidates.items(),
        key=lambda kv: (
            kv[1]["median_completion_tokens_per_sec"],
            kv[1]["median_total_tokens_per_sec"],
            -kv[0],  # à débit égal, préférer C plus bas
        ),
    )

    return (
        best_c,
        best_v["median_completion_tokens_per_sec"],
        best_v["median_total_tokens_per_sec"],
    )


def build_run_summary(
    *,
    start_time: float,
    end_time: float,
    all_results: list[RequestResult],
    all_summaries: list[PlateauSummary],
) -> RunSummary:
    duration = max(end_time - start_time, 0.001)
    ok = [r for r in all_results if r.ok]
    fail = [r for r in all_results if not r.ok]

    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    total_tokens = sum(r.total_tokens for r in ok)

    best_c, best_out_tps, best_total_tps = choose_best_concurrency(all_summaries)

    return RunSummary(
        duration_sec=duration,
        total_requests=len(all_results),
        success=len(ok),
        fail=len(fail),
        success_rate=len(ok) / len(all_results) if all_results else 0.0,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        avg_prompt_tokens_per_sec=prompt_tokens / duration,
        avg_completion_tokens_per_sec=completion_tokens / duration,
        avg_total_tokens_per_sec=total_tokens / duration,
        optimal_concurrency=best_c,
        best_completion_tokens_per_sec=best_out_tps,
        best_total_tokens_per_sec=best_total_tps,
    )


def print_final_report(
    *,
    run_summary: RunSummary,
    all_summaries: list[PlateauSummary],
    all_results: list[RequestResult],
) -> None:
    print("\n" + "=" * 72)
    print("📊 RAPPORT FINAL")
    print("=" * 72)

    print(f"Durée réelle:              {run_summary.duration_sec:.1f}s")
    print(f"Requêtes totales:          {run_summary.total_requests}")
    print(f"Succès:                    {run_summary.success} ({run_summary.success_rate*100:.1f}%)")
    print(f"Échecs:                    {run_summary.fail}")

    print("\n🔤 TOKENS")
    print(f"Input total:               {run_summary.total_prompt_tokens:>12d}")
    print(f"Output total:              {run_summary.total_completion_tokens:>12d}")
    print(f"Total tokens:              {run_summary.total_tokens:>12d}")
    print(f"Input tok/s moyen:         {run_summary.avg_prompt_tokens_per_sec:>12.1f}")
    print(f"Output tok/s moyen:        {run_summary.avg_completion_tokens_per_sec:>12.1f}")
    print(f"Total tok/s moyen:         {run_summary.avg_total_tokens_per_sec:>12.1f}")

    print("\n🏆 MEILLEUR POINT STABLE")
    print(f"Concurrence optimale:      {run_summary.optimal_concurrency}")
    print(f"Best output tok/s médian:  {run_summary.best_completion_tokens_per_sec:.1f}")
    print(f"Best total tok/s médian:   {run_summary.best_total_tokens_per_sec:.1f}")

    lats = [r.latency_ms for r in all_results]
    if lats:
        print("\n⏱️ LATENCES")
        print(f"Moyenne:                   {sum(lats)/len(lats):.0f} ms")
        print(f"P50:                       {percentile(lats, 0.50):.0f} ms")
        print(f"P90:                       {percentile(lats, 0.90):.0f} ms")
        print(f"P95:                       {percentile(lats, 0.95):.0f} ms")
        print(f"P99:                       {percentile(lats, 0.99):.0f} ms")
        print(f"Max:                       {max(lats):.0f} ms")

    # Erreurs globales
    errors: dict[str, int] = {}
    for r in all_results:
        if not r.ok:
            errors[r.error or "unknown"] = errors.get(r.error or "unknown", 0) + 1

    if errors:
        print("\n❌ ERREURS")
        for err, count in sorted(errors.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {count:>5d} × {err}")

    grouped = group_by_concurrency(all_summaries)
    best_c = run_summary.optimal_concurrency
    best_out = max((v["median_completion_tokens_per_sec"] for v in grouped.values()), default=1.0)

    print("\n📊 PERFORMANCE PAR CONCURRENCE")
    print("─" * 90)
    print(
        f"{'C':>4} | {'Mes.':>4} | {'SR méd.':>8} | "
        f"{'Out tok/s méd.':>14} | {'Out max':>10} | "
        f"{'Total tok/s méd.':>16} | Score"
    )
    print("─" * 90)

    for c, v in sorted(grouped.items()):
        med_out = v["median_completion_tokens_per_sec"]
        bar_len = int((med_out / max(best_out, 1.0)) * 20)
        bar = "█" * bar_len
        marker = " ★ BEST" if c == best_c else ""
        print(
            f"{c:>4d} | "
            f"{v['measurements']:>4d} | "
            f"{v['median_success_rate']*100:>7.1f}% | "
            f"{med_out:>14.1f} | "
            f"{v['max_completion_tokens_per_sec']:>10.1f} | "
            f"{v['median_total_tokens_per_sec']:>16.1f} | "
            f"{bar}{marker}"
        )


def write_json_report(
    path: str,
    *,
    run_summary: RunSummary,
    all_summaries: list[PlateauSummary],
    all_results: list[RequestResult],
    args: argparse.Namespace,
) -> None:
    grouped = group_by_concurrency(all_summaries)

    payload = {
        "mode": "token_throughput_maximizer",
        "config": {
            "api_url": API_URL,
            "model": MODEL,
            "stress_duration": args.duration,
            "warmup_sec": args.warmup,
            "discovery_plateau_sec": args.discovery_plateau,
            "max_concurrency": args.max_concurrency,
            "request_timeout": args.timeout,
            "max_tokens": args.max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "prompt_mode": args.prompt_mode,
            "min_success_rate": MIN_SUCCESS_RATE,
            "step_size": STEP_SIZE,
            "exploit_probe_interval": EXPLOIT_PROBE_INTERVAL,
        },
        "summary": asdict(run_summary),
        "by_concurrency": grouped,
        "plateaus": [asdict(s) for s in all_summaries],
        "requests": [asdict(r) for r in all_results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv_report(path: str, all_results: list[RequestResult]) -> None:
    fields = [
        "ok",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "status",
        "error",
        "phase",
        "concurrency",
        "started_at",
        "ended_at",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))


# ============================================================
# CLI / Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Maximise token throughput on an OpenAI-compatible Chat Completions API."
    )

    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--api-key", default=API_KEY)
    p.add_argument("--model", default=MODEL)

    p.add_argument("--duration", type=int, default=STRESS_DURATION)
    p.add_argument("--warmup", type=int, default=WARMUP_SEC)
    p.add_argument("--discovery-plateau", type=int, default=DISCOVERY_PLATEAU_SEC)
    p.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT)

    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument(
        "--prompt-mode",
        choices=["long_output", "short_output", "mixed"],
        default="long_output",
        help="long_output maximise surtout completion_tokens/sec.",
    )

    p.add_argument("--output-json", default=OUTPUT_JSON)
    p.add_argument("--output-csv", default=OUTPUT_CSV)

    return p.parse_args()


async def main() -> None:
    global API_URL, API_KEY, MODEL, MAX_CONCURRENCY, REQUEST_TIMEOUT, MAX_TOKENS

    args = parse_args()

    API_URL = args.api_url
    API_KEY = args.api_key
    MODEL = args.model
    MAX_CONCURRENCY = args.max_concurrency
    REQUEST_TIMEOUT = args.timeout
    MAX_TOKENS = args.max_tokens

    if not MODEL:
        print("⚠️  MODEL vide. Beaucoup de serveurs OpenAI-compatible exigent un model.")
    if not API_URL:
        raise SystemExit("API_URL manquant.")

    print("🧠 LLM TOKEN THROUGHPUT MAXIMIZER")
    print(f"   API:              {API_URL}")
    print(f"   Model:            {MODEL}")
    print(f"   Durée:            {args.duration}s")
    print(f"   Warmup:           {args.warmup}s")
    print(f"   Discovery plateau:{args.discovery_plateau}s")
    print(f"   Max concurrency:  {args.max_concurrency}")
    print(f"   Request timeout:  {args.timeout}s")
    print(f"   max_tokens:       {args.max_tokens}")
    print(f"   prompt mode:      {args.prompt_mode}")
    print(f"   JSON:             {args.output_json}")
    print(f"   CSV:              {args.output_csv}")

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    connector = aiohttp.TCPConnector(
        limit=0,               # pas de limite aiohttp globale
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    start = time.monotonic()
    global_deadline = start + args.duration
    prompt_counter = 0

    all_summaries: list[PlateauSummary] = []
    all_results: list[RequestResult] = []

    async with aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        connector=connector,
    ) as session:
        # Warmup
        prompt_counter = await warmup(
            session,
            prompt_mode=args.prompt_mode,
            prompt_counter=prompt_counter,
        )

        # Discovery
        best_c, disc_summaries, disc_results, prompt_counter = await discovery(
            session,
            prompt_mode=args.prompt_mode,
            prompt_counter=prompt_counter,
            global_deadline=global_deadline,
        )
        all_summaries.extend(disc_summaries)
        all_results.extend(disc_results)

        # Exploitation
        remaining = global_deadline - time.monotonic()
        if remaining > 2:
            final_c, exp_summaries, exp_results, prompt_counter = await exploitation(
                session,
                initial_concurrency=best_c,
                prompt_mode=args.prompt_mode,
                prompt_counter=prompt_counter,
                global_deadline=global_deadline,
            )
            all_summaries.extend(exp_summaries)
            all_results.extend(exp_results)

    end = time.monotonic()

    run_summary = build_run_summary(
        start_time=start,
        end_time=end,
        all_results=all_results,
        all_summaries=all_summaries,
    )

    print_final_report(
        run_summary=run_summary,
        all_summaries=all_summaries,
        all_results=all_results,
    )

    write_json_report(
        args.output_json,
        run_summary=run_summary,
        all_summaries=all_summaries,
        all_results=all_results,
        args=args,
    )
    write_csv_report(args.output_csv, all_results)

    print(f"\n💾 Rapport JSON: {args.output_json}")
    print(f"💾 Requêtes CSV: {args.output_csv}")
    print("🏁 Terminé")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Interrompu par l'utilisateur.")
