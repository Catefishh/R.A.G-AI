# Aurelia Support FAQ

Frequently asked questions handled by the Nimbus Forge support hub.

### How do I reset a leaked API key?

Run `forge-sdk keys rotate --key-id <id>` or use the dashboard under
**Settings → API Keys → Rotate**. Rotation is instant and the old key stops
working within **60 seconds**. Leaked `aur_live_` keys should always be rotated,
never just disabled.

### Why am I getting error AUR-429?

`AUR-429` means you have exceeded your edition's per-stream ingestion rate
limit. On the Team edition the limit is 10,000 messages/sec. The fix is either
to batch messages using `UnifiedFrame.batch()` or to upgrade to Enterprise.

### What is the "cold frame" warning?

A **cold frame** warning (`AUR-W12`) appears when a stream has received no data
for more than **15 minutes**. Aurelia keeps the stream open but stops billing
for it until data resumes. It is informational, not an error.

### Does Aurelia store my raw data?

By default Aurelia stores normalised UnifiedFrames for the retention window of
your edition. Raw, pre-normalisation payloads are **discarded immediately**
after normalisation unless you enable **Raw Capture**, an Enterprise-only add-on.

### How do I contact support?

- Team edition: email support@nimbusforge.example, response within 1 business day.
- Enterprise: your dedicated Slack Connect channel, 1-hour SLA during business
  hours in your contracted region.
- Community (Starter): the public forum at community.nimbusforge.example.
