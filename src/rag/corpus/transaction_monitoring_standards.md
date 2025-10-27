# Transaction Monitoring Standards

## Country-Level Rules
| Country | Velocity Limit (tx/day) | Avg Spend (USD) | High-Value Threshold |
|----------|------------------------|-----------------|----------------------|
| US | 100 | 60 | 500 |
| MX | 80 | 45 | 400 |
| AR | 60 | 35 | 350 |
| BR | 70 | 40 | 400 |
| CL | 75 | 42 | 400 |

## Monitoring Goals
- Detect regional anomalies in transaction velocity.
- Capture month-end surges without over-flagging.
- Align fraud thresholds with regional benchmarks.

## Data Sources
Logs from the Payments Platform are ingested hourly into the Data Warehouse.
