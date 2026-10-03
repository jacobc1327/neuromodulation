"""Command-line entry point: ``neuromod <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings


def _ml(args):
    from neuromod.pipeline import run_ml_pipeline

    run_ml_pipeline(out_dir=args.out, n=args.n, seed=args.seed, n_iter=args.n_iter,
                    figures=not args.no_figures)


def _simulate(args):
    from neuromod.data.synthetic import simulate_cohort

    cohort = simulate_cohort(n=args.n, seed=args.seed)
    cohort.data.to_csv(args.out, index=False)
    print(f"Wrote {len(cohort.data)} SYNTHETIC patients -> {args.out}")


def _rag(args):
    from neuromod.rag.pipeline import RTMSResearchAssistant

    assistant = RTMSResearchAssistant(k=args.k)
    ans = assistant.run(" ".join(args.question), mode=args.mode, strict=args.strict)
    if args.json:
        print(json.dumps(ans.__dict__, indent=2, default=str))
    else:
        print(ans.to_markdown())


def _rag_eval(args):
    from neuromod.rag.evaluate import evaluate_grounding, evaluate_retrieval, load_eval
    from neuromod.rag.pipeline import RTMSResearchAssistant

    assistant = RTMSResearchAssistant()
    qs = load_eval()
    result = {"retrieval": evaluate_retrieval(assistant.index, qs, k=args.k),
              "grounding": evaluate_grounding(assistant, qs)}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.figure:
        import pandas as pd

        from neuromod.viz.plots import corpus_overview

        corpus_overview(pd.DataFrame([s.__dict__ for s in assistant.index.studies]), args.figure)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neuromod", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ml = sub.add_parser("ml", help="run the full ML pipeline (synthetic cohort -> models -> SHAP)")
    ml.add_argument("--out", default="reports")
    ml.add_argument("--n", type=int, default=2000)
    ml.add_argument("--seed", type=int, default=7)
    ml.add_argument("--n-iter", type=int, default=25, help="hyper-parameter search iterations")
    ml.add_argument("--no-figures", action="store_true")
    ml.set_defaults(func=_ml)

    sim = sub.add_parser("simulate", help="write a synthetic veteran cohort to CSV")
    sim.add_argument("--n", type=int, default=2000)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--out", default="synthetic_cohort.csv")
    sim.set_defaults(func=_simulate)

    for name, mode, help_ in [("ask", "ask", "evidence synthesis for a question"),
                              ("hypotheses", "hypotheses", "generate testable hypotheses"),
                              ("design", "design", "draft a study-design brief")]:
        r = sub.add_parser(name, help=help_)
        r.add_argument("question", nargs="+")
        r.add_argument("--k", type=int, default=6, help="number of studies to retrieve")
        r.add_argument("--strict", action="store_true", help="drop unsupported sentences")
        r.add_argument("--json", action="store_true")
        r.set_defaults(func=_rag, mode=mode)

    ev = sub.add_parser("rag-eval", help="evaluate retrieval and citation grounding")
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--out", default=None)
    ev.add_argument("--figure", default=None, help="also write a corpus overview PNG here")
    ev.set_defaults(func=_rag_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
