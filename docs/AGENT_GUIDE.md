# Single-agent research assistant

## What it demonstrates

The assistant is a production-minded orchestration layer around the existing ML system. It accepts
English/Hinglish questions, selects one supported intent, invokes only approved tools, retrieves
methodology evidence from repository documentation, and returns an auditable report.

## Supported workflows

| Workflow | Example prompt | Executed tools |
|---|---|---|
| Assessment | `Is profile ka risk score explain karo` | prediction, conservative explanation, report |
| Comparison | `Dono scenarios compare karke farak batao` | scenario comparison, report |
| Methodology | `0.4890 aur 0.4694 threshold ka difference kya hai?` | document retrieval, report |

## Why this is a single agent

One orchestrator owns planning and execution. Specialist behavior lives in small tools rather than
separate agents. This is easier to test, cheaper to operate, and appropriate for three bounded
workflows. A multi-agent design would add coordination failure modes without improving the result.

## Guardrails

- Pydantic validates all structured records and forbids unknown fields.
- The LLM, when enabled, can select only one of three enum intents.
- Application code owns the tool mapping; the LLM cannot invent or execute a tool.
- The scikit-learn bundle alone calculates probabilities.
- Methodology retrieval reads only four approved repository documents.
- Unknown prompts fall back to methodology rather than model inference.
- Every response identifies the model version, demo status, tools, evidence, and disclaimer.
- No conversation or respondent profile is persisted by the application.

## Recruiter demo sequence

1. Ask `How was the model validated?` and open the evidence/tool trace.
2. Ask for one profile's score and show that the prediction comes from `predict_risk`.
3. Compare two scenarios and explain why the difference is not a causal effect.
4. Download the JSON report.
5. Show `scripts/evaluate_agent.py` and the green CI run.

## Honest limitations

The public model is trained on synthetic data and demonstrates software behavior only. Retrieval is
small-scale lexical search, not semantic retrieval over a large corpus. The optional LLM improves
language routing but is not allowed to calculate risk, provide medical advice, or change model
outputs. This system requires governance, recalibration, privacy review, and human oversight before
any real-world study.
