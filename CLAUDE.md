# FlowShield — Project Instructions

## Project

FlowShield is an Autonomous Payment Reliability & Revenue Protection Agent for the Razorpay Buildathon 2026.

Track:
AI Revenue Recovery

Deadline:
September 5, 2026

## Core Objective

FlowShield detects payment degradation, identifies likely root causes, estimates revenue at risk, evaluates recovery strategies, applies deterministic safety policies, executes bounded recovery actions, verifies outcomes, and maintains an audit trail.

Core pipeline:

Payment Events
→ Payment Health Engine
→ Anomaly Detection
→ Incident Detection
→ AI Root Cause Analysis
→ Revenue Impact Analysis
→ Counterfactual Simulation
→ Policy / Guardrails
→ Recovery Execution
→ Outcome Verification
→ Audit Trail

## Core Principle

AI RECOMMENDS.
POLICY ENGINE AUTHORIZES.
EXECUTION LAYER ACTS.
OUTCOME ENGINE MEASURES.

The LLM must never directly authorize financial actions.

## AI

Primary reasoning model:

NVIDIA Nemotron 3 Ultra.

Use an abstraction:

ReasoningProvider
├── NemotronProvider
└── MockReasoningProvider

The application must remain functional when Nemotron is unavailable.

Do not hard-code the application to a single model provider.

## ML

ML/statistical components should handle:

1. Payment-health anomaly detection
2. Recovery probability prediction
3. Revenue-impact estimation

Do not use an LLM where deterministic/statistical/ML logic is more appropriate.

## Synthetic Data

Because production payment data is unavailable, development and evaluation will use synthetic payment events.

Synthetic data must be clearly labeled.

Synthetic scenarios should include:

1. Bank/payment-rail degradation
2. Regional degradation
3. Latency spike
4. Merchant/system degradation
5. Isolated failures
6. Normal traffic

Synthetic data must be reproducible using fixed random seeds.

Do not fabricate evaluation results.

## Recovery Actions

Initially support:

- RETRY
- ALTERNATIVE_ROUTE
- CUSTOMER_NUDGE
- ESCALATE
- STOP

Do not implement unnecessary action types unless justified.

## Safety

The deterministic policy engine must enforce limits such as:

- Maximum retry attempts
- Maximum automatic recovery amount
- Minimum confidence threshold
- Maximum recovery window
- Duplicate-action prevention

Never allow the LLM to bypass policy.

## Evaluation

The primary business metric is:

RECOVERED REVENUE

Compare FlowShield against a simple baseline such as:

"Retry every eligible failed payment once."

Measure:

- Recovery rate
- Recovered revenue
- Revenue protected
- Recovery uplift
- Unnecessary recovery attempts
- Escalations
- Duplicate actions prevented
- Unsafe actions blocked
- False interventions

Evaluation must use a reproducible test set.

## Architecture

Prefer a modular architecture.

Avoid unnecessary microservices.

Avoid overengineering.

Use clear interfaces between:

- ingestion
- health metrics
- anomaly detection
- incidents
- reasoning
- recovery
- simulation
- policy
- execution
- verification
- audit
- evaluation

## Development Rules

Before making significant architectural changes:

1. Explain the reason.
2. Check whether the change conflicts with existing architecture.
3. Prefer the smallest change that solves the problem.

Do not rewrite working components unnecessarily.

Do not create giant files.

Use type hints.

Write tests for important logic.

Never hard-code secrets.

Use environment variables for API keys and credentials.

Never commit .env files or secrets.

## Razorpay

Razorpay integration must use test/sandbox functionality where available.

Never claim production payment execution.

Clearly distinguish:

- real Razorpay test-mode behavior
- synthetic simulation
- mocked behavior

## Git

Make focused commits.

Use meaningful commit messages.

Do not make massive unrelated commits.

## Collaboration

ChatGPT is acting as:

- Lead Architect
- ML Advisor
- Code Reviewer
- Research Advisor
- Hackathon Strategist

Claude is the primary coding agent and repository owner.

When receiving implementation instructions from ChatGPT, implement only the requested phase unless explicitly instructed otherwise.

After completing a task:

1. Run tests.
2. Run relevant validation.
3. Report what changed.
4. Report commands executed.
5. Report test results.
6. Report known issues.
7. Stop and wait for the next task.

## Current Phase

PHASE 1:

Repository foundation
+
canonical payment schema
+
synthetic payment-event generator
+
controlled incident scenarios
+
data validation
+
basic analysis
+
tests
+
documentation

Do NOT implement Nemotron, recovery logic, dashboard, Razorpay integration, or advanced ML until explicitly instructed.