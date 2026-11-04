# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Use GitHub's
private vulnerability reporting feature for this repository. Include the affected
route or feature, reproduction steps, expected impact, and any relevant logs with
tokens or personal information removed.

## Supported version

The latest version deployed from the `main` branch is supported. Older commits and
locally modified deployments are not covered by this policy.

## Scope

Useful reports include unsafe file import behavior, cross-site scripting, exposure
of private deployment values, bypasses of experiment workload limits, and ways to
execute code through custom strategies. The custom strategy builder is intended to
accept only the documented rule schema and never arbitrary Python or JavaScript.
