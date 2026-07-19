# Security Policy

## Supported versions

NeuCoDe is pre-1.0 research software. Security fixes are applied to the latest
release on the `main` branch only.

| Version | Supported |
| ------- | --------- |
| latest `main` | yes |
| older tags    | no  |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue.

- Preferred: use GitHub's **"Report a vulnerability"** button under the
  repository's *Security* tab (private vulnerability reporting).
- Alternatively, email **fischerbe98484@th-nuernberg.de** with the details.

Please include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected versions/commit, and
- any suggested remediation.

You can expect an initial acknowledgement within a reasonable time. Please give
us a chance to release a fix before any public disclosure.

## Scope notes

NeuCoDe controls physical hardware (motors, gimbals) and parses telemetry from
serial devices. When evaluating impact, keep in mind:

- The hardware-communication layer trusts data received over the serial link.
  Only connect to firmware and devices you control.
- Generated firmware model headers and exported model artifacts are produced
  from your own training data and are not sandboxed.
- The optional BrainChip Akida path pulls proprietary third-party packages; report
  vulnerabilities in those packages to BrainChip, not here.
