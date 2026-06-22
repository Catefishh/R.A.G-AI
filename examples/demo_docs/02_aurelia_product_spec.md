# Aurelia Platform — Product Specification (v3.2)

Aurelia is Nimbus Forge's real-time data orchestration platform. This document
describes the v3.2 release, shipped on **2026-04-14**.

## What Aurelia does

Aurelia ingests sensor and telemetry streams from robotics fleets, normalises
them into a common schema called **UnifiedFrame**, and routes them to
downstream consumers (dashboards, model training jobs, alerting).

## Editions and pricing

Aurelia is sold in three editions. Prices are per month, billed annually, in USD.

| Edition    | Price/mo | Streams | Retention | Support        |
|------------|----------|---------|-----------|----------------|
| Starter    | $49      | 5       | 7 days    | Community       |
| Team       | $399     | 50      | 30 days   | Email, 1 bus. day |
| Enterprise | Custom   | Unlimited | 1 year+ | Dedicated SLA   |

The Starter edition is **free for verified academic robotics labs**.

## Key limits (v3.2)

- Maximum **UnifiedFrame** message size: **2 MB**.
- Maximum ingestion rate per stream on the Team edition: **10,000 messages/sec**.
- Default data retention on Team: **30 days** (configurable up to 90 on request).
- Aurelia regions in v3.2: **Sydney, Frankfurt, and Oregon**.

## The Aurelia SDK

The official SDK is called **forge-sdk** and is available for Python and Rust.
A connection is opened with an API key prefixed `aur_live_` (production) or
`aur_test_` (sandbox). Sandbox keys are rate-limited to 100 messages/sec.
