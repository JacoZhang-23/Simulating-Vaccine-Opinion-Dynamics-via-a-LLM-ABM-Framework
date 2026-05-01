# Subsample Network Generation Report (BA Model)

**Generated at**: 2026-02-27 10:52:48.723816

## Parameters

- **Random Seed**: 42
- **SM Target Avg Degree**: 50
- **SM Age Range**: 13-64
- **Method**: Barabási-Albert model + Edge pruning

## Population Statistics

- **Total Population**: 12,360
- **Nodes in Network**: 11,639 (94.2%)
- **Isolated Nodes**: 721 (5.8%)

## SM Network Statistics (BA Model)

| Metric | Value | Notes |
|--------|-------|-------|
| Eligible Population (13-64岁) | 7,789 | Total SM-eligible users |
| Nodes in Graph | 7,789 | 100.0% of eligible |
| Isolated SM Users | 0 | 0.0% of eligible |
| Edges | 194,100 | Total connections |
| Avg Degree (Graph) | 49.84 | Among connected nodes |
| Avg Degree (Total) | 49.84 | Target: 50 |
| Median Degree | 35 | - |
| Max Degree | 739 | - |
| Min Degree | 25 | - |
| Density | 0.006400 | - |
| Connected Components | 1 | - |
| Largest Component | 7,789 nodes | - |

## Complete Network Statistics

| Network Type | Edges | Percentage |
|--------------|-------|------------|
| hh | 13,075 | 5.6% |
| wk | 18,795 | 8.1% |
| sc | 5,083 | 2.2% |
| dc | 1,844 | 0.8% |
| sm | 194,100 | 83.3% |
| **Total** | 232,652 | 100.0% |

## Overall Network Properties

- **Total Edges**: 232,652
- **Average Degree (Graph)**: 39.98
- **Average Degree (Total Pop)**: 37.65
- **Connected Components**: 385

## Method: Barabási-Albert Model + Edge Pruning

The SM network is generated using a **scale-free model**:

1. **Generate BA network**: Use Barabási-Albert model with preferential attachment
2. **Edge pruning**: Randomly remove edges to reach target avg degree (50)
3. **Result**: Scale-free network with adjustable density

**Advantages**:
- Scale-free property (power-law degree distribution)
- Adjustable average degree
- Efficient generation for large networks

