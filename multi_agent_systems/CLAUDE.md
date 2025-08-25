# IRONFORGE Multi-Agent Systems

## 🎯 Corrected Liquidity Progression Analysis System

**Status:** Production-ready multi-agent coordination  
**Purpose:** Analyze what pre-existing liquidity targets get taken after 40% archaeological zone touches

### **Agent Architecture:**

1. **Pre-Structure Inventory Agent** (`pre_structure_inventory_agent.py`)
   - **Role:** Market structure archaeologist
   - **Function:** Catalogs session highs/lows, daily extremes, and existing FVGs that existed BEFORE 40% touches
   - **Output:** Complete pre-existing liquidity target inventory

2. **Target Progression Tracker Agent** (`target_progression_tracker_agent.py`)  
   - **Role:** Liquidity hunting specialist
   - **Function:** Monitors which pre-existing targets get taken AFTER 40% touches
   - **Focus:** Current session vs next session completion timing

3. **Statistical Prediction Generator Agent** (`statistical_prediction_generator_agent.py`)
   - **Role:** Probability synthesizer  
   - **Function:** Converts tracking data into predictive intelligence
   - **Output:** Actionable probability forecasts for live trading

### **Coordination Framework:**

**Data Flow:** Pre-Structure Inventory → Target Progression → Statistical Predictions  
**Results Storage:**
- Inventories: `/pre_structure_inventories/`
- Tracking: `/target_progression_tracking/`  
- Forecasts: `/liquidity_progression_forecasts/`

### **Key Discovery:**

After 40% archaeological zone touches:
- ✅ **Session highs/lows:** 100% completion when targeted
- ✅ **Daily progression:** Session→daily liquidity targeting
- ❌ **FVG redeliveries:** 0% success rate (gaps don't get filled)

**Timing:** Peak completion window 63-141 minutes post-touch

### **TMux Coordination:**

Use `tmux_multiagent_setup.sh` for coordinated 3-agent analysis  
Refer to `TMUX_MULTIAGENT_USAGE.md` for operational procedures

---

**Critical Framework:** Based on `40_PERCENT_LIQUIDITY_PROGRESSION_HYPOTHESIS.md` - prevents communication errors and ensures correct analysis focus.