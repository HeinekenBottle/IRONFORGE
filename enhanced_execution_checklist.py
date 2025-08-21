#!/usr/bin/env python3
"""
Enhanced Execution Checklist: Macro-Archaeological AM Trading
Real-time checklist integrating ICT methodology with IRONFORGE intelligence
"""

from datetime import datetime, time
from typing import Dict, List, Any
from dataclasses import dataclass
from macro_archaeological_framework import MacroArchaeologicalFramework
import json

@dataclass
class ChecklistItem:
    """Individual checklist item with completion tracking"""
    task: str
    completed: bool = False
    timestamp: datetime = None
    notes: str = ""
    priority: str = "normal"  # "critical", "high", "normal", "low"

class EnhancedExecutionChecklist:
    """
    Comprehensive execution checklist for AM scalping with archaeological context
    Integrates all components: pre-open, Gauntlet, macro timing, risk management
    """
    
    def __init__(self):
        self.framework = MacroArchaeologicalFramework()
        self.checklist_items: List[ChecklistItem] = []
        self.session_context = {}
        
    def generate_preopen_checklist(self, trading_date: str) -> List[ChecklistItem]:
        """Generate pre-open preparation checklist (08:45–09:29 ET)"""
        
        preopen_items = [
            ChecklistItem(
                task="📊 HTF bias locked (VI/CE map says 'longs only' or 'shorts only')",
                priority="critical"
            ),
            ChecklistItem(
                task="🗺️ Pre-open archaeological zones mapped from prior session completion",
                priority="critical"
            ),
            ChecklistItem(
                task="📍 London H/L marked and integrated with archaeological levels",
                priority="high"
            ),
            ChecklistItem(
                task="🌏 Asia H/L marked and integrated with archaeological levels", 
                priority="high"
            ),
            ChecklistItem(
                task="🌙 Midnight open level marked",
                priority="normal"
            ),
            ChecklistItem(
                task="📈 Prior day imbalances/CE levels identified",
                priority="high"
            ),
            ChecklistItem(
                task="🎯 Primary confluence levels ranked by archaeological significance",
                priority="high"
            ),
            ChecklistItem(
                task="⚠️ News risk assessed (NFP 8:30 ET, ISM 10:00 ET hazard zones)",
                priority="critical"
            ),
            ChecklistItem(
                task="📱 Trading platform setup with archaeological zone alerts",
                priority="normal"
            ),
            ChecklistItem(
                task="💰 Position sizing calculated (base 1x, archaeological 1.5x multiplier)",
                priority="normal"
            )
        ]
        
        return preopen_items
    
    def generate_execution_checklist(self, current_time: datetime) -> List[ChecklistItem]:
        """Generate real-time execution checklist"""
        
        current_time_only = current_time.time()
        
        # Determine active trading window
        in_primary = time(9, 35) <= current_time_only <= time(9, 58)
        in_macro = time(9, 50) <= current_time_only <= time(10, 10)
        in_variant = time(10, 20) <= current_time_only <= time(10, 40)
        in_hazard = time(9, 55) <= current_time_only <= time(10, 5)  # NFP hazard zone
        
        execution_items = []
        
        # Window-specific items
        if in_primary:
            execution_items.extend([
                ChecklistItem(
                    task="🎯 PRIMARY WINDOW (9:35-9:58): Monitor for cash open digestion patterns",
                    priority="critical"
                ),
                ChecklistItem(
                    task="👁️ Watch for sweep → reclaim of Gauntlet FVG (lowest first SSIB)",
                    priority="critical"
                ),
                ChecklistItem(
                    task="📏 Measure Gauntlet CE proximity to archaeological zones (<7.55pts = 2x confidence)",
                    priority="high"
                )
            ])
        
        if in_macro:
            execution_items.extend([
                ChecklistItem(
                    task="🔄 MACRO WINDOW (9:50-10:10): Assess orbital phase (setup/entry/extension/completion)",
                    priority="critical"
                ),
                ChecklistItem(
                    task="📊 Determine follow-through vs fade based on orbital phase context",
                    priority="high"
                ),
                ChecklistItem(
                    task="⚡ Execute only during entry/extension phases, avoid completion",
                    priority="critical"
                )
            ])
        
        if in_variant:
            execution_items.append(
                ChecklistItem(
                    task="🔄 VARIANT WINDOW (10:20-10:40): Only if 10:00 shock didn't break structure",
                    priority="normal"
                )
            )
        
        if in_hazard:
            execution_items.append(
                ChecklistItem(
                    task="⚠️ HAZARD ZONE (9:55-10:05): Extreme caution, defensive positioning",
                    priority="critical"
                )
            )
        
        # Universal execution items
        execution_items.extend([
            ChecklistItem(
                task="🎯 Gauntlet FVG identified with CE level marked",
                priority="critical"
            ),
            ChecklistItem(
                task="✅ Sweep confirmation: liquidity raid below mapped sell-side",
                priority="critical"
            ),
            ChecklistItem(
                task="📈 Reclaim confirmation: immediate reclaim of Gauntlet CE",
                priority="critical"
            ),
            ChecklistItem(
                task="🚀 M1 displacement confirmation away from CE",
                priority="high"
            ),
            ChecklistItem(
                task="📊 Structure shift confirmation (micro HH/HL for longs)",
                priority="high"
            ),
            ChecklistItem(
                task="🎯 Archaeological confluence check (±7.55pts for 2x confidence)",
                priority="high"
            )
        ])
        
        return execution_items
    
    def generate_risk_management_checklist(self, entry_price: float, direction: str) -> List[ChecklistItem]:
        """Generate risk management checklist for active position"""
        
        risk_items = [
            ChecklistItem(
                task=f"🛑 Stop loss set 1R behind sweep extreme",
                priority="critical"
            ),
            ChecklistItem(
                task="🎯 First partial (+10 handles) - 50% position scale",
                priority="high"
            ),
            ChecklistItem(
                task="🎯 Second partial (+20-30 handles) - 30% position scale",
                priority="high"
            ),
            ChecklistItem(
                task="📈 Trail stop to Gauntlet midpoint/CE after first target",
                priority="normal"
            ),
            ChecklistItem(
                task="⏱️ Monitor 10-12 minute expansion rule",
                priority="high"
            ),
            ChecklistItem(
                task="❌ Invalidation: full body close back through CE",
                priority="critical"
            ),
            ChecklistItem(
                task="🏺 Archaeological zone proximity monitoring (<7.55pts = exit signal)",
                priority="high"
            ),
            ChecklistItem(
                task="🔄 Orbital completion phase scaling (7-10 minutes into macro window)",
                priority="normal"
            )
        ]
        
        return risk_items
    
    def generate_session_close_checklist(self) -> List[ChecklistItem]:
        """Generate end-of-session review checklist"""
        
        close_items = [
            ChecklistItem(
                task="📊 All positions closed or managed to breakeven",
                priority="critical"
            ),
            ChecklistItem(
                task="📈 Trade journal updated with archaeological confluence notes",
                priority="high"
            ),
            ChecklistItem(
                task="🎯 Gauntlet setups reviewed for archaeological proximity correlation",
                priority="normal"
            ),
            ChecklistItem(
                task="⏰ Macro window timing effectiveness assessed",
                priority="normal"
            ),
            ChecklistItem(
                task="🔄 Orbital phase performance analyzed",
                priority="normal"
            ),
            ChecklistItem(
                task="🏺 Archaeological zone accuracy validated (Theory B precision)",
                priority="high"
            ),
            ChecklistItem(
                task="📋 Session statistics updated (confluence rate, avg confidence)",
                priority="normal"
            ),
            ChecklistItem(
                task="🎯 Next session archaeological zones prepared",
                priority="normal"
            )
        ]
        
        return close_items
    
    def create_comprehensive_checklist(self, trading_date: str, current_time: datetime, 
                                     active_position: bool = False, 
                                     entry_details: Dict = None) -> Dict[str, List[ChecklistItem]]:
        """Create comprehensive checklist for current trading state"""
        
        checklist = {
            'preopen': self.generate_preopen_checklist(trading_date),
            'execution': self.generate_execution_checklist(current_time),
            'session_close': self.generate_session_close_checklist()
        }
        
        if active_position and entry_details:
            checklist['risk_management'] = self.generate_risk_management_checklist(
                entry_details.get('entry_price', 0),
                entry_details.get('direction', 'long')
            )
        
        return checklist
    
    def display_checklist(self, checklist: Dict[str, List[ChecklistItem]], 
                         section: str = None):
        """Display formatted checklist"""
        
        sections_to_show = [section] if section else checklist.keys()
        
        for section_name in sections_to_show:
            if section_name not in checklist:
                continue
                
            print(f"\n{'='*50}")
            print(f"📋 {section_name.upper().replace('_', ' ')} CHECKLIST")
            print(f"{'='*50}")
            
            items = checklist[section_name]
            
            # Group by priority
            critical_items = [item for item in items if item.priority == 'critical']
            high_items = [item for item in items if item.priority == 'high']
            normal_items = [item for item in items if item.priority == 'normal']
            
            for priority_group, group_name in [(critical_items, 'CRITICAL'), 
                                              (high_items, 'HIGH'), 
                                              (normal_items, 'NORMAL')]:
                if group_name == 'CRITICAL' and critical_items:
                    print(f"\n🚨 {group_name} PRIORITY:")
                elif group_name == 'HIGH' and high_items:
                    print(f"\n⚡ {group_name} PRIORITY:")
                elif group_name == 'NORMAL' and normal_items:
                    print(f"\n📝 {group_name} PRIORITY:")
                
                for item in priority_group:
                    status = "✅" if item.completed else "⬜"
                    print(f"   {status} {item.task}")
                    if item.notes:
                        print(f"      💭 {item.notes}")
    
    def mark_item_complete(self, checklist: Dict[str, List[ChecklistItem]], 
                          section: str, task_partial: str, notes: str = ""):
        """Mark checklist item as complete"""
        
        if section not in checklist:
            return False
        
        for item in checklist[section]:
            if task_partial.lower() in item.task.lower():
                item.completed = True
                item.timestamp = datetime.now()
                item.notes = notes
                return True
        
        return False
    
    def get_completion_status(self, checklist: Dict[str, List[ChecklistItem]]) -> Dict[str, float]:
        """Get completion percentage for each section"""
        
        status = {}
        
        for section_name, items in checklist.items():
            completed_count = sum(1 for item in items if item.completed)
            total_count = len(items)
            completion_pct = (completed_count / total_count * 100) if total_count > 0 else 0
            status[section_name] = completion_pct
        
        return status

def demo_enhanced_checklist():
    """Demo the enhanced execution checklist"""
    print("🎯 Enhanced Execution Checklist Demo")
    print("=" * 50)
    
    checklist_manager = EnhancedExecutionChecklist()
    
    # Generate comprehensive checklist
    trading_date = "2025-08-07"
    current_time = datetime(2025, 8, 7, 9, 52)  # 9:52 AM ET
    
    checklist = checklist_manager.create_comprehensive_checklist(
        trading_date, current_time
    )
    
    # Display execution checklist (most relevant during trading)
    checklist_manager.display_checklist(checklist, 'execution')
    
    # Show completion status
    print(f"\n📊 COMPLETION STATUS:")
    status = checklist_manager.get_completion_status(checklist)
    for section, pct in status.items():
        print(f"   {section.replace('_', ' ').title()}: {pct:.0f}%")
    
    # Demo marking item complete
    print(f"\n✅ Marking HTF bias as complete...")
    checklist_manager.mark_item_complete(
        checklist, 'preopen', 'HTF bias', 
        "Bullish bias confirmed via daily VI confluence"
    )
    
    updated_status = checklist_manager.get_completion_status(checklist)
    print(f"   Pre-open: {updated_status['preopen']:.0f}% complete")

if __name__ == "__main__":
    demo_enhanced_checklist()