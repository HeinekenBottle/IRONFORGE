# Contract Compliance Enforcer

Validate IRONFORGE golden invariants, enforce session boundaries, and check performance contracts.

## Usage

```python
from agents.contract_compliance_enforcer.agent import create_contract_compliance_enforcer

agent = create_contract_compliance_enforcer()
session = {
  "events": ["Expansion", "Retracement"],
  "edge_intents": ["TEMPORAL_NEXT", "CONTEXT"],
  "node_feature_dim": 51,
  "edge_feature_dim": 20,
}
print(agent.validate_golden_invariants(session))
```

## Contracts
- Events: 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- Edge intents: 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- Features: 51D nodes (f0-f50), 20D edges (e0-e19)
- HTF rule: Last-closed only; no intra-candle
- Session isolation: No cross-session edges
