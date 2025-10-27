# Fraud Detection Guidelines

## Core Criteria
Transactions should be flagged as *potentially fraudulent* if they meet any of these conditions:
1. Amount **exceeds $500**.
2. Occurs between **2:00 AM and 5:00 AM** local time.
3. Exceeds **50 transactions per day** by the same user.
4. Originates from a new device not previously associated with the account.

## Review Workflow
- Automated systems tag the transaction with a risk score.
- Analysts verify suspicious cases manually before blocking.
- Confirmed frauds are reported to the Security & Compliance team.

## Notes
Always document false positives to improve detection thresholds.
