# Security Policy

## Supported versions

RoboInspect is pre-1.0 and under active development. Security fixes are applied to
the latest released version on a best-effort basis.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Instead,
report them privately via [GitHub's private vulnerability reporting](https://github.com/robocurve/roboinspect/security/advisories/new)
("Report a vulnerability" under the repository's Security tab).

Include a description, reproduction steps, and the potential impact. We will
acknowledge your report, work with you on a fix, and credit you (if you wish)
once a fix is released.

## Safety note

RoboInspect can command physical robots. The framework provides an error taxonomy
and an approver safety gate, but **operators are responsible for hardware
safety** — workspace limits, e-stops, and supervision during real-world runs.
Treat any policy as untrusted until validated in simulation.
