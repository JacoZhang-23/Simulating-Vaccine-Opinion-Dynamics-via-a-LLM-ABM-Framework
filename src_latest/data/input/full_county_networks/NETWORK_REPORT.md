# Full County Network Generation Report

**Generated at**: 2026-02-27 10:39:11.910587

## Parameters

- **Random Seed**: 42
- **Workplace Size Range**: (10, 50)
- **SM Target Avg Degree**: 50
- **SM Age Range**: 13-64

## Population Statistics

- **Total Population**: 127,657
- **Nodes in Network**: 123,119 (96.4%)
- **Isolated Nodes**: 4,538 (3.6%)

## Network Statistics

| Network Type | Edges | Percentage |
|--------------|-------|------------|
| hh | 141,038 | 6.0% |
| wk | 91,341 | 3.9% |
| sc | 48,839 | 2.1% |
| dc | 16,150 | 0.7% |
| sm | 2,060,425 | 87.4% |
| **Total** | 2,357,793 | 100.0% |

## Overall Network Properties

- **Total Edges**: 2,356,814
- **Average Degree**: 38.29
- **Network Density**: 0.000311
- **Connected Components**: 1520

## SM Network Details (BA Model)

| Metric | Value |
|--------|-------|
| Eligible Population | 82,442 |
| Nodes in Graph | 82,442 |
| Isolated Nodes | 0 |
| Edges | 2,060,425 |
| Avg Degree (Graph) | 49.98 |
| Avg Degree (Total) | 49.98 |
| Density | 0.000606 |
| Components | 1 |

## Method Details

### Workplace Network (WK)
- **Method**: Small-world network (Newman-Watts-Strogatz)
- **Workplace size**: 10-50 people
- **Network parameters**: k=4, p=0.3

### Social Media Network (SM)
- **Method**: Barabási-Albert model + Edge pruning
- **Target avg degree**: 50
- **Age range**: 13-64
- **Process**: Generate BA network → Randomly remove edges to reach target degree

