"""
Baseline 4: STP-style self-play (Self-play Through the Pipeline).

The conjecturer is trained with a REINFORCE reward that targets the
"barely provable" difficulty frontier: conjectures proven in ≤ frontier_k
steps earn +1, proven but in > frontier_k steps earn 0, not proven earn -0.5.

This directly mirrors STP (arXiv:2502.00212, ICML 2025) but within the
AutoConjecture Peano arithmetic framework.

Key difference from AutoConjecture Phase 5:
  - Prover: heuristic BFS only (no RL prover / PPO on tactic selection)
  - Generator update: REINFORCE policy gradient on frontier reward
    (not supervised imitation on proven theorems)
"""
from __future__ import annotations

import time
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from ..logic.axioms import get_all_axioms
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..prover.proof_engine import Proof
from ..training.parallel_prover import ParallelHeuristicProver
from ..generation.neural_generator import NeuralConjectureGenerator
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..generation.novelty import NoveltyScorer
from ..models.transformer_generator import TransformerGenerator
from ..models.tokenizer import ExpressionTokenizer
from ..models.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner


class STPBaselineRunner(BaselineRunner):
    """
    STP-style self-play: frontier-reward REINFORCE on the generator.

    The generator is trained to produce statements at the "barely provable"
    difficulty frontier, adaptively tracking what the prover can handle.
    """

    @property
    def name(self) -> str:
        return "stp"

    def setup(self, config: dict) -> None:
        self.device = config.get("device", "cpu")
        var_names = config.get("var_names", ["x", "y", "z", "w"])

        self.kb = KnowledgeBase(axioms=get_all_axioms())

        self.tokenizer = ExpressionTokenizer(max_length=128, var_names=var_names)

        self.gen_model = TransformerGenerator(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.get("gen_d_model", 256),
            nhead=config.get("gen_nhead", 8),
            num_layers=config.get("gen_num_layers", 6),
        ).to(self.device)

        self.neural_gen = NeuralConjectureGenerator(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            device=self.device,
            temperature=config.get("temperature", 1.2),  # Slightly higher for exploration
            top_k=config.get("top_k", 50),
        )

        self.random_gen = RandomConjectureGenerator(
            min_complexity=config.get("min_complexity", 6),
            max_complexity=config.get("max_complexity", 20),
            var_names=var_names,
            seed=config.get("seed", 42),
        )

        # Supervised trainer for periodic imitation updates
        gen_trainer_cfg = GeneratorTrainingConfig(
            learning_rate=config.get("gen_lr", 1e-4),
            batch_size=config.get("gen_batch_size", 32),
            warmup_steps=config.get("gen_warmup_steps", 500),
            device=self.device,
        )
        self.gen_trainer = GeneratorTrainer(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            config=gen_trainer_cfg,
            knowledge_base=self.kb,
        )

        n_workers = config.get("parallel_workers", 2)
        self.parallel_prover = ParallelHeuristicProver(
            max_workers=n_workers,
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            timeout_per_proof=config.get("timeout_per_proof", 30.0),
        )

        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=config.get("max_similar", 20))
        self.novelty_scorer = NoveltyScorer()
        self.metrics = ComparisonMetrics(self.name)

        # STP-specific hyperparameters
        self.frontier_k: int = config.get("frontier_k", 20)
        self.reinforce_lr: float = config.get("reinforce_lr", 1e-5)
        self.reinforce_optimizer = torch.optim.Adam(
            self.gen_model.parameters(), lr=self.reinforce_lr
        )
        # EMA baseline for variance reduction
        self._reward_baseline: float = 0.0
        self._ema_alpha: float = config.get("ema_alpha", 0.05)

        self._batch_size: int = config.get("batch_size", 10)
        self._neural_ratio: float = config.get("neural_ratio", 0.7)
        # How often to do a REINFORCE update (every N cycles)
        self._reinforce_interval: int = config.get("reinforce_interval", 10)
        # How often to mix in a supervised imitation update
        self._supervised_interval: int = config.get("supervised_interval", 50)

        self._attempted: int = 0
        self._proved: int = 0
        self._cycles: int = 0

        # Buffer: (token_seq_list, reward) pairs for REINFORCE
        self._reinforce_buffer: List[Tuple[List[int], float]] = []

    def run_for(
        self,
        wall_clock_budget_seconds: float,
        snapshot_interval_seconds: float = 60.0,
    ) -> List[ComparisonSnapshot]:
        snapshots: List[ComparisonSnapshot] = []
        t_start = time.time()
        t_last_snap = t_start

        while True:
            now = time.time()
            if (now - t_start) >= wall_clock_budget_seconds:
                break

            self._cycles += 1

            # ── generate with log-prob tracking ───────────────────────────
            n_neural = int(self._batch_size * self._neural_ratio)
            n_random = self._batch_size - n_neural

            conjectures = []
            token_seqs = []  # parallel list, None for random-generated conjectures

            if n_neural > 0:
                neural_conjs, neural_seqs = self._generate_with_tokens(n_neural)
                conjectures.extend(neural_conjs)
                token_seqs.extend(neural_seqs)
            if n_random > 0:
                rand_conjs = self.random_gen.generate(n_random)
                conjectures.extend(rand_conjs)
                token_seqs.extend([None] * len(rand_conjs))

            # ── filter ────────────────────────────────────────────────────
            filtered_conjs = []
            filtered_seqs = []
            for c, seq in zip(conjectures, token_seqs):
                if c is None:
                    continue
                if not self.complexity_est.is_well_formed(c):
                    continue
                if self.novelty_scorer.score(c) < 0.3:
                    continue
                if not self.diversity_filter.should_keep(c):
                    continue
                if self.kb.contains(c):
                    continue
                filtered_conjs.append(c)
                filtered_seqs.append(seq)

            if not filtered_conjs:
                continue

            # ── prove ─────────────────────────────────────────────────────
            kb_stmts = self.kb.get_all_statements()
            kb_strs = [str(s) for s in kb_stmts]
            expr_strs = [str(c) for c in filtered_conjs]
            self._attempted += len(filtered_conjs)

            try:
                results = self.parallel_prover.prove_batch(expr_strs, kb_strs)
            except Exception:
                results = self.parallel_prover.prove_batch_sequential(expr_strs, kb_strs)

            # ── assign STP frontier rewards and collect KB additions ───────
            for c, seq, res in zip(filtered_conjs, filtered_seqs, results):
                reward = self._frontier_reward(res["success"], res["proof_length"])

                # Update EMA baseline
                self._reward_baseline = (
                    (1 - self._ema_alpha) * self._reward_baseline
                    + self._ema_alpha * reward
                )

                # Buffer neural sequences for REINFORCE
                if seq is not None:
                    self._reinforce_buffer.append((seq, reward))

                # Add proven theorems to KB
                if res["success"] and not self.kb.contains(c):
                    complexity = self.complexity_est.estimate(c)
                    stub = Proof(statement=c, steps=[], result=ProofResult.SUCCESS)
                    added = self.kb.add_theorem(c, stub, complexity, 0, self._attempted)
                    if added:
                        self._proved += 1
                        self.novelty_scorer.add(c)

            # ── REINFORCE update ──────────────────────────────────────────
            if self._cycles % self._reinforce_interval == 0 and self._reinforce_buffer:
                self._reinforce_update()
                self._reinforce_buffer.clear()

            # ── Occasional supervised imitation (stability regulariser) ───
            if self._cycles % self._supervised_interval == 0 and self.kb.size() >= 5:
                self._supervised_update()

            # ── snapshot ──────────────────────────────────────────────────
            now = time.time()
            if (now - t_last_snap) >= snapshot_interval_seconds:
                snap = self.metrics.snapshot(
                    self.kb, now - t_start, self._attempted, self._proved
                )
                snapshots.append(snap)
                print(snap.summary_line(), flush=True)
                t_last_snap = now

        final_snap = self.metrics.snapshot(
            self.kb, time.time() - t_start, self._attempted, self._proved
        )
        snapshots.append(final_snap)
        return snapshots

    def _frontier_reward(self, success: bool, proof_length: int) -> float:
        """
        Frontier reward: incentivise conjectures at the barely-provable boundary.
          Proved in ≤ frontier_k steps : +1.0
          Proved but in > frontier_k steps: +0.0  (too easy or lucky)
          Not proved                       : -0.5  (too hard)
        """
        if success:
            return 1.0 if proof_length <= self.frontier_k else 0.0
        return -0.5

    def _generate_with_tokens(
        self, n: int
    ) -> Tuple[List, List[Optional[List[int]]]]:
        """
        Generate n conjectures using the neural model, returning both the
        decoded expressions and the raw token sequences (needed for REINFORCE).
        """
        from ..logic.expressions import Expression

        self.neural_gen.eval_mode()

        batch_size = min(n, 32)
        num_batches = (n + batch_size - 1) // batch_size

        all_exprs: List = []
        all_seqs: List[Optional[List[int]]] = []

        with torch.no_grad():
            for bi in range(num_batches):
                cur = min(batch_size, n - bi * batch_size)
                seqs = self.gen_model.generate(
                    batch_size=cur,
                    max_length=self.tokenizer.max_length,
                    temperature=self.neural_gen.temperature,
                    top_k=self.neural_gen.top_k,
                    top_p=self.neural_gen.top_p,
                    sos_token_id=self.tokenizer.sos_id,
                    eos_token_id=self.tokenizer.eos_id,
                    pad_token_id=self.tokenizer.pad_id,
                    device=self.device,
                )
                for seq in seqs:
                    seq_list = seq.cpu().tolist()
                    expr = self.tokenizer.decode_tokens(seq_list)
                    all_exprs.append(expr)
                    all_seqs.append(seq_list if expr is not None else None)

        return all_exprs, all_seqs

    def _reinforce_update(self):
        """
        REINFORCE policy gradient update on the generator.

        For each (token_seq, reward) in the buffer:
          1. Run a forward pass on the token sequence to obtain log-probabilities.
          2. Use the baseline-subtracted reward to scale the log-prob sum.
          3. Accumulate gradients and step once.
        """
        if not self._reinforce_buffer:
            return

        self.gen_model.train()
        self.reinforce_optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid = 0

        for token_seq, reward in self._reinforce_buffer:
            # Baseline-subtracted reward
            advantage = reward - self._reward_baseline
            if abs(advantage) < 1e-8:
                continue

            try:
                # Prepare token tensor: (seq_len,)
                tokens = torch.tensor(token_seq, dtype=torch.long, device=self.device)
                if tokens.shape[0] < 2:
                    continue

                input_tokens = tokens[:-1].unsqueeze(1)   # (seq_len-1, 1)
                target_tokens = tokens[1:].unsqueeze(1)   # (seq_len-1, 1)

                src = torch.empty(
                    0, 1, dtype=torch.long, device=self.device
                )
                logits = self.gen_model(
                    src=src,
                    tgt=input_tokens,
                    tgt_mask=None,
                    src_key_padding_mask=None,
                    tgt_key_padding_mask=None,
                )  # (seq_len-1, 1, vocab_size)

                # Log-prob of each generated token
                log_probs = F.log_softmax(logits, dim=-1)  # (seq_len-1, 1, vocab)
                token_log_probs = log_probs.gather(
                    2, target_tokens.unsqueeze(-1).expand_as(log_probs[:, :, :1])
                    .clamp(0, log_probs.shape[2] - 1)
                )  # Safer gather

                # Actually: gather the specific token log-prob
                vocab_size = log_probs.shape[2]
                tgt_clamped = target_tokens.clamp(0, vocab_size - 1)  # (seq_len-1, 1)
                seq_log_prob = log_probs.gather(
                    2, tgt_clamped.unsqueeze(2)
                ).squeeze(2).squeeze(1)  # (seq_len-1,)

                # Skip pad tokens in loss
                pad_mask = (target_tokens.squeeze(1) != self.tokenizer.pad_id).float()
                seq_log_prob = (seq_log_prob * pad_mask).sum()

                # REINFORCE loss: -advantage * log_prob  (we want to maximise reward)
                loss_i = -advantage * seq_log_prob
                total_loss = total_loss + loss_i
                valid += 1

            except Exception:
                continue

        if valid > 0:
            total_loss = total_loss / valid
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 1.0)
                self.reinforce_optimizer.step()
                print(
                    f"  [{self.name}] REINFORCE update | "
                    f"n={valid} | loss={total_loss.item():.4f} | "
                    f"baseline={self._reward_baseline:.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{self.name}] REINFORCE backward failed: {e}", flush=True)

        self.gen_model.eval()

    def _supervised_update(self):
        """Occasional supervised imitation to stabilise training."""
        recent_count = min(50, self.kb.size())
        if recent_count < 5:
            return
        recent = self.kb.get_all_theorems()[-recent_count:]
        exprs = [t.statement for t in recent]
        self.neural_gen.train_mode()
        try:
            loss = self.gen_trainer._train_batch(exprs)
            print(f"  [{self.name}] supervised update loss={loss:.4f}", flush=True)
        except Exception:
            pass
        finally:
            self.neural_gen.eval_mode()

    def get_final_kb(self) -> KnowledgeBase:
        return self.kb
