# Senior MLOps Engineer, Take-Home Assignment

PipeSense Analytics predicts leak risk for water utility pipe networks. The batch pipeline in
`starter_kit/` is how it runs today: the whole team shares one big conda environment, the stages
pass files to each other in a shared folder, and there is a first attempt at a Dockerfile. It
works, but it will not survive growth.

We want this pipeline containerized, reproducible, and ready to serve hundreds of utility
customers that request analyses on demand. The requirements are below. How you meet them is up
to you, and your reasoning weighs as much as your code. Where the requirements are silent, do
not ask us: make an assumption, write it in DECISIONS.md, and proceed.

## Rules

- You have 7 days. Requirements are graded independently, so a partial submission is fine.
- AI tools are allowed, we use them daily. Say how you used them in DECISIONS.md.
- After you submit there is a ~45 minute interview. We review your pipeline together and walk
  through your code. Expect questions about the machine learning side as well.
- Assume the company runs on AWS, jobs take from 2 minutes to several hours, the platform team
  is two people, and live customers depend on the current system, so changes must be able to
  land incrementally.
- You do not need an AWS account. Everything runs locally with Docker and docker-compose. Where
  a real deployment detail matters, describe it in writing.

## Deliverables

A git repo (zip or private link) with a real commit history: the starter kit evolved,
DECISIONS.md (findings from your code review, assumptions, decisions with the alternatives you
weighed, known debt, AI usage), DESIGN.md for requirement 5, and a top-level README.md that
works on a clean machine with only Docker and compose. We do read the git history. Scoring: 100
points plus up to 15 bonus.

## Requirement 1: a reproducible environment (15 pts)

Give the pipeline a runtime environment that contains only what it actually needs, and make sure
a teammate can recreate it exactly.

## Requirement 2: the pipeline runs in containers (20 pts)

Make the stages run in containers you would happily put in production. Start by writing down in
DECISIONS.md what is wrong with the Dockerfile we have today. You decide how the stages run one
after the other: files, a queue, an orchestration tool, anything you can defend. Think about
failure too: when a stage dies midway, the stages that already finished should not have to run
again, and DECISIONS.md should say how you achieved that. Write down how
you version your images and what you do when a bad one reaches production. The tests must pass
inside your image, or inside a test variant of it. Finally, write the code so that a stage could
move to a different orchestration system without a rewrite.

## Requirement 3: a development and a production environment (10 pts)

The same image runs in a development setup and in a production-like setup, selected by
configuration only. Show how a change travels from one to the other. Keep it config driven; we
do not expect any CI/CD setup here.

## Requirement 4: a managed model lifecycle (25 pts)

Models are versioned in a registry (we use MLflow). A freshly trained model must not become the
serving model automatically: it passes a promotion gate, and the decision is auditable. You
define what winning means. Every batch of predictions records which model version produced it.

## Requirement 5: a design for running at scale (20 pts, no code)

Write DESIGN.md (2 pages max, one diagram): how you would run this pipeline on AWS, event
driven, for hundreds of customers requesting jobs on demand. Open-source components (a broker,
for example) are welcome alongside AWS services. Address how a job gets triggered
and executed, what happens when things fail, how you migrate there without disrupting live
customers, how a team of two operates it, and roughly what it costs. What you cover, and what
you leave out, is part of what we evaluate.

## DECISIONS.md (10 pts)

Graded on its own. Several problems are in the starter kit on purpose, including on the ML
side; we expect your review to find them. This file drives the interview.

## Bonus: a working miniature of your design (+15 pts, optional)

Only attempt this if the requirements are solid. Locally, a queue of your choice receives
`{"job_id": ..., "utility": "fairport"}` and that triggers your containerized pipeline. Status
transitions are observable somewhere that is not a file you refresh by hand. Make one job fail
on purpose and show how your system deals with it; what should happen is your call, and the
call is part of the grade.
