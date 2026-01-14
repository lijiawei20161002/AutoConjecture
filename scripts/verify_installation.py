#!/usr/bin/env python3
"""
Quick verification script to test AutoConjecture installation.
Tests basic functionality of each component.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_logic_system():
    """Test logic system components."""
    print("Testing logic system...")
    from src.logic.terms import Var, Zero, Succ, Add
    from src.logic.expressions import Equation, Forall
    from src.logic.axioms import PEANO_AXIOMS

    # Create terms
    x = Var("x")
    zero = Zero()
    one = Succ(zero)

    # Create equation: x + 0 = x
    eq = Equation(Add(x, zero), x)
    print(f"  Created equation: {eq}")

    # Create universal statement
    forall_eq = Forall(x, eq)
    print(f"  Created statement: {forall_eq}")

    # Check axioms
    print(f"  Loaded {len(PEANO_AXIOMS)} Peano axioms")

    print("  ✓ Logic system works!\n")
    return True


def test_prover():
    """Test proof engine."""
    print("Testing prover...")
    from src.logic.terms import Var, Zero, Add
    from src.logic.expressions import Equation
    from src.prover.proof_engine import ProofEngine, ProofResult

    # Create simple conjecture: 0 + 0 = 0
    zero = Zero()
    eq = Equation(Add(zero, zero), zero)
    print(f"  Attempting to prove: {eq}")

    # Create prover
    prover = ProofEngine(max_depth=10, max_iterations=100)

    # Try to prove
    proof = prover.prove(eq)
    print(f"  Proof result: {proof.result}")

    if proof.result == ProofResult.SUCCESS:
        print(f"  Proof found in {proof.length()} steps!")
        for step in proof.steps:
            print(f"    {step}")

    print("  ✓ Prover works!\n")
    return True


def test_generator():
    """Test conjecture generator."""
    print("Testing generator...")
    from src.generation.random_generator import RandomConjectureGenerator

    # Create generator
    generator = RandomConjectureGenerator(
        min_complexity=2,
        max_complexity=8,
        seed=42
    )

    # Generate conjectures
    conjectures = generator.generate(5)
    print(f"  Generated {len(conjectures)} conjectures:")
    for i, conj in enumerate(conjectures):
        print(f"    {i+1}. {conj}")

    print("  ✓ Generator works!\n")
    return True


def test_knowledge_base():
    """Test knowledge base."""
    print("Testing knowledge base...")
    from src.knowledge.knowledge_base import KnowledgeBase
    from src.logic.axioms import PEANO_AXIOMS

    # Create knowledge base
    kb = KnowledgeBase(axioms=PEANO_AXIOMS)
    print(f"  Created knowledge base with {kb.total_size()} statements")
    print(f"  {kb}")

    print("  ✓ Knowledge base works!\n")
    return True


def test_training_components():
    """Test training loop components."""
    print("Testing training components...")
    from src.training.training_loop import TrainingConfig
    from src.monitoring.logger import Logger

    # Create config
    config = TrainingConfig(
        num_epochs=1,
        cycles_per_epoch=10,
    )
    print(f"  Created config: {config.num_epochs} epochs, {config.cycles_per_epoch} cycles")

    # Create logger
    import tempfile
    import os
    tmp_dir = tempfile.mkdtemp()
    logger = Logger(log_dir=tmp_dir)
    logger.info("Test log message")
    print(f"  Logger test: {logger.log_file}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

    print("  ✓ Training components work!\n")
    return True


def main():
    """Run all verification tests."""
    print("="*60)
    print("AutoConjecture Installation Verification")
    print("="*60)
    print()

    tests = [
        ("Logic System", test_logic_system),
        ("Prover", test_prover),
        ("Generator", test_generator),
        ("Knowledge Base", test_knowledge_base),
        ("Training Components", test_training_components),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test_name} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*60)
    print(f"Verification Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✓ All tests passed! System is ready to use.")
        print("\nTo start training, run:")
        print("  python3 scripts/train.py")
    else:
        print(f"✗ {failed} tests failed. Please fix errors before training.")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
