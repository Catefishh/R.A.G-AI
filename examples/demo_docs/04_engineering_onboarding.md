# Engineering Onboarding Guide

Welcome to the Aurelia platform team. This guide covers the parts of our stack a
new engineer needs in their first week.

## Architecture in one paragraph

Aurelia is a set of Rust services behind a single gateway called **Skyline**.
Skyline authenticates requests, then hands streams to the **Normaliser**, which
produces UnifiedFrames. Normalised frames go onto an internal log called
**Tideway** (our own append-only log, not Kafka). Consumers subscribe to Tideway
partitions. Metadata lives in PostgreSQL; UnifiedFrames live in object storage.

## Local development

- Clone the monorepo `nimbus-forge/aurelia`.
- Run `just bootstrap` to install toolchains (Rust 1.83, Python 3.12).
- `just dev` starts Skyline, the Normaliser, and a single-node Tideway locally.
- Tests: `just test`. The end-to-end suite is `just e2e` and needs Docker.

## Deployment cadence

We deploy to production **every Tuesday and Thursday** at 10:00 NZST. Hotfixes
can go out any time with two approvals. There is a **hard freeze** on deploys in
the last week of each quarter.

## On-call

On-call rotates weekly. The primary on-call owns Skyline and the Normaliser; the
secondary owns Tideway and storage. Pages route through the tool we call
**Watchtower**. The on-call runbook lives at `docs/runbooks/oncall.md` in the
monorepo.

## Who to ask

- Skyline / auth: **Mara Lindqvist**
- Normaliser: **Tcomás Beites**
- Tideway / storage: **Priya Raman** (also co-founder)
- Anything billing-related: the Lisbon office
