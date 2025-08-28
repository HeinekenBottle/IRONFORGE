#!/usr/bin/env python3
"""
IRONFORGE Research Framework Validator
Ensures research-agnostic, configuration-driven methodology is always used

This validator prevents teams from:
- Hardcoding pattern assumptions (40% zones, specific event families)
- Bypassing agent coordination for complex research
- Skipping statistical validation and quality gates
- Using ad-hoc research approaches instead of systematic methods
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import yaml

logger = logging.getLogger(__name__)


class ResearchFrameworkViolation(Exception):
    """Raised when research framework principles are violated"""
    pass


class ResearchFrameworkValidator:
    """Validates that IRONFORGE research follows configuration-driven, agent-coordinated methodology"""
    
    def __init__(self):
        self.hardcoded_patterns = {
            # Pattern assumptions that should be configurable
            'percentage_zones': [
                r'\b40%?\s*zone',
                r'forty\s*percent',
                r'0\.40?\s*level',
                r'archaeological.*40',
            ],
            'event_families': [
                r'FVG.*family',
                r'liquidity.*family',
                r'expansion.*family',
                r'hardcoded.*event.*type',
            ],
            'temporal_assumptions': [
                r'temporal.*non.*locality.*always',
                r'events.*know.*final',
                r'dimensional.*relationship.*fixed',
            ],
            'pattern_hardcoding': [
                r'if.*zone.*==.*40',
                r'zone_percentage.*=.*0\.4',
                r'archaeological.*level.*=.*40',
            ]
        }
        
        self.required_research_elements = {
            'configuration': ['research_question', 'hypothesis_parameters', 'discovery_method'],
            'statistical': ['validation_method', 'quality_threshold', 'significance_testing'],
            'agents': ['agent_roles', 'coordination_method'],
            'methodology': ['data_source', 'tgat_config', 'authenticity_threshold']
        }
        
    def validate_research_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a research file follows framework principles"""
        
        results = {
            'file': str(file_path),
            'compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            content = file_path.read_text()
            
            # Check for hardcoded pattern assumptions
            violations = self._check_hardcoded_patterns(content)
            if violations:
                results['violations'].extend(violations)
                results['compliant'] = False
            
            # Check for configuration-driven approach
            config_issues = self._check_configuration_driven(content, file_path)
            if config_issues:
                results['warnings'].extend(config_issues)
            
            # Check for agent coordination usage
            agent_issues = self._check_agent_coordination(content)
            if agent_issues:
                results['recommendations'].extend(agent_issues)
                
            # Check for statistical rigor
            stats_issues = self._check_statistical_rigor(content)
            if stats_issues:
                results['violations'].extend(stats_issues)
                results['compliant'] = False
                
        except Exception as e:
            results['violations'].append(f"Validation error: {e}")
            results['compliant'] = False
            
        return results
        
    def _check_hardcoded_patterns(self, content: str) -> List[str]:
        """Check for hardcoded pattern assumptions"""
        violations = []
        
        for category, patterns in self.hardcoded_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(
                        f"HARDCODED {category.upper()}: Found '{matches[0]}' - "
                        f"Should use configuration parameter instead"
                    )
        
        # Check for specific anti-patterns
        anti_patterns = [
            (r'zone_percentage\s*=\s*0\.4', "Hardcoded 40% zone - use config parameter"),
            (r'if.*archaeological.*zone', "Hardcoded archaeological logic - use configurable discovery"),
            (r'FVG.*redelivery.*always', "Assumed FVG behavior - let TGAT discover patterns"),
            (r'temporal.*non.*locality.*true', "Assumed temporal relationship - test hypothesis instead"),
        ]
        
        for pattern, message in anti_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(f"ANTI-PATTERN: {message}")
                
        return violations
    
    def _check_configuration_driven(self, content: str, file_path: Path) -> List[str]:
        """Check if research is configuration-driven"""
        warnings = []
        
        # Look for configuration files or parameters
        has_config = any([
            'research_question' in content,
            'hypothesis_parameters' in content,
            'config.yaml' in content,
            'load_config' in content,
            '.yml' in content,
            'ConfigurableResearch' in content
        ])
        
        if not has_config:
            warnings.append(
                "No configuration detected - research should be driven by "
                "hypothesis configuration, not hardcoded assumptions"
            )
            
        # Check for direct hardcoding vs parameterization
        hardcode_indicators = [
            r'percentage_levels\s*=\s*\[.*40.*\]',
            r'event_types\s*=\s*\[.*FVG.*\]',
            r'zone_levels\s*=\s*\[.*23.*\]'
        ]
        
        for indicator in hardcode_indicators:
            if re.search(indicator, content):
                warnings.append(
                    "Found hardcoded research parameters - consider using "
                    "configurable research_question and hypothesis_parameters"
                )
                
        return warnings
    
    def _check_agent_coordination(self, content: str) -> List[str]:
        """Check for proper agent coordination usage"""
        recommendations = []
        
        # Check for agent imports/usage
        agent_usage = any([
            'data-scientist' in content,
            'knowledge-architect' in content, 
            'adjacent-possible-linker' in content,
            'scrum-master' in content,
            'Task(' in content and 'agent' in content,
            'multi_agent' in content
        ])
        
        # Check for complexity indicators that should use agents
        complexity_indicators = [
            r'hypothesis.*test',
            r'statistical.*analysis',
            r'pattern.*discover',
            r'cross.*session.*learn',
            r'research.*question',
            r'multiple.*timeframe',
        ]
        
        has_complexity = any(re.search(ind, content, re.IGNORECASE) for ind in complexity_indicators)
        
        if has_complexity and not agent_usage:
            recommendations.append(
                "Complex research detected - consider using agent coordination "
                "(data-scientist, knowledge-architect, adjacent-possible-linker) "
                "for systematic analysis and knowledge management"
            )
            
        # Check for manual coordination vs systematic
        manual_indicators = [
            r'manual.*analysis',
            r'ad.*hoc.*research',
            r'quick.*test',
            r'simple.*check'
        ]
        
        has_manual = any(re.search(ind, content, re.IGNORECASE) for ind in manual_indicators)
        
        if has_manual:
            recommendations.append(
                "Manual research approach detected - IRONFORGE provides "
                "sophisticated agent coordination for systematic analysis"
            )
            
        return recommendations
    
    def _check_statistical_rigor(self, content: str) -> List[str]:
        """Check for statistical rigor requirements"""
        violations = []
        
        # Required statistical elements for research
        required_stats = [
            r'statistical.*significance',
            r'confidence.*interval',
            r'p.*value',
            r'permutation.*test',
            r'quality.*threshold',
            r'authenticity.*87',
        ]
        
        has_stats = any(re.search(stat, content, re.IGNORECASE) for stat in required_stats)
        
        # Check for research patterns that require statistical validation
        research_patterns = [
            r'pattern.*discover',
            r'hypothesis.*test',
            r'correlation.*analy',
            r'clustering.*event',
            r'temporal.*relationship',
        ]
        
        has_research = any(re.search(pattern, content, re.IGNORECASE) for pattern in research_patterns)
        
        if has_research and not has_stats:
            violations.append(
                "STATISTICAL RIGOR MISSING: Research patterns detected without "
                "statistical validation - must include significance testing, "
                "confidence intervals, and quality thresholds (87% authenticity)"
            )
            
        # Check for TGAT usage without quality gates
        has_tgat = 'tgat' in content.lower() or 'discovery' in content.lower()
        has_quality = any([
            'authenticity' in content,
            'quality_threshold' in content,
            '87%' in content,
            '0.87' in content
        ])
        
        if has_tgat and not has_quality:
            violations.append(
                "TGAT QUALITY GATES MISSING: TGAT usage detected without "
                "authenticity thresholds - must enforce 87% quality standard"
            )
            
        return violations
        
    def validate_research_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate research configuration file follows framework principles"""
        
        results = {
            'config_file': str(config_path),
            'compliant': True,
            'missing_elements': [],
            'good_practices': [],
            'framework_score': 0
        }
        
        try:
            if config_path.suffix in ['.yml', '.yaml']:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path) as f:
                    config = json.load(f)
                    
            # Check for required research elements
            for category, elements in self.required_research_elements.items():
                for element in elements:
                    if self._find_nested_key(config, element):
                        results['good_practices'].append(f"✅ {category}: {element}")
                        results['framework_score'] += 1
                    else:
                        results['missing_elements'].append(f"❌ {category}: {element}")
                        
            # Specific framework validations
            framework_checks = [
                ('research_question', "Research question clearly defined"),
                ('hypothesis_parameters', "Hypothesis parameters configurable"),
                ('agents', "Agent coordination specified"),
                ('statistical', "Statistical validation configured"),
                ('quality_threshold', "Quality thresholds defined"),
            ]
            
            for key, description in framework_checks:
                if self._find_nested_key(config, key):
                    results['good_practices'].append(f"✅ {description}")
                    
        except Exception as e:
            results['missing_elements'].append(f"Config validation error: {e}")
            results['compliant'] = False
            
        # Determine compliance
        total_possible = sum(len(elements) for elements in self.required_research_elements.values())
        compliance_rate = results['framework_score'] / total_possible if total_possible > 0 else 0
        
        results['compliant'] = compliance_rate >= 0.7  # 70% compliance threshold
        results['compliance_rate'] = compliance_rate
        
        return results
    
    def _find_nested_key(self, config: dict, key: str) -> bool:
        """Find key in nested dictionary structure"""
        if isinstance(config, dict):
            if key in config:
                return True
            for value in config.values():
                if self._find_nested_key(value, key):
                    return True
        return False
        
    def generate_compliance_report(self, validation_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive compliance report"""
        
        total_files = len(validation_results)
        compliant_files = sum(1 for r in validation_results if r['compliant'])
        
        report = f"""
# IRONFORGE Research Framework Compliance Report

## Summary
- **Total Files Analyzed**: {total_files}
- **Compliant Files**: {compliant_files}/{total_files} ({compliant_files/total_files*100:.1f}%)
- **Framework Adherence**: {'✅ GOOD' if compliant_files/total_files >= 0.8 else '⚠️ NEEDS IMPROVEMENT'}

## Violations by Category

"""
        
        all_violations = []
        all_recommendations = []
        
        for result in validation_results:
            all_violations.extend(result.get('violations', []))
            all_recommendations.extend(result.get('recommendations', []))
            
        # Categorize violations
        violation_categories = {}
        for violation in all_violations:
            category = violation.split(':')[0]
            violation_categories.setdefault(category, []).append(violation)
            
        for category, violations in violation_categories.items():
            report += f"\n### {category}\n"
            for violation in set(violations):  # Remove duplicates
                report += f"- {violation}\n"
                
        report += f"""

## Recommendations for Team

### Immediate Actions Required:
"""
        
        for rec in set(all_recommendations[:5]):  # Top 5 unique recommendations
            report += f"- {rec}\n"
            
        report += f"""

### Framework Best Practices:

1. **Configuration-First Research**
   - Always start with research_question and hypothesis_parameters
   - Use configurable percentage_levels instead of hardcoding 40%
   - Define discovery_method and validation_method explicitly

2. **Agent Coordination**  
   - Use data-scientist agent for statistical analysis
   - Use knowledge-architect agent for cross-session learning
   - Use adjacent-possible-linker agent for creative pattern connections
   - Use scrum-master agent for complex research project management

3. **Statistical Rigor**
   - Always include significance testing (p-values, confidence intervals)  
   - Enforce 87% authenticity thresholds for TGAT discovery
   - Use permutation testing for pattern validation
   - Document quality gates and compliance rates

4. **Research Methodology**
   - Let TGAT discover patterns rather than assuming event families exist
   - Test percentage zones as hypotheses (20%, 60%, 85%) not assumptions
   - Use temporal relationships as research questions, not hardcoded rules
   - Apply agent coordination for systematic analysis

## Next Steps

To improve framework compliance:

1. **Review flagged files** and convert hardcoded assumptions to configuration
2. **Implement agent coordination** for complex research workflows  
3. **Add statistical validation** to all pattern discovery work
4. **Create research configurations** that define hypotheses explicitly
5. **Use IRONFORGE's professional research capabilities** systematically

---

*This report ensures IRONFORGE is used as a flexible, research-agnostic platform 
rather than hardcoded to specific pattern assumptions.*
"""
        
        return report


def validate_ironforge_research(directory: str = ".") -> None:
    """Main validation function for IRONFORGE research framework compliance"""
    
    validator = ResearchFrameworkValidator()
    results = []
    
    # Find research files
    research_files = []
    base_path = Path(directory)
    
    # Common research file patterns
    patterns = [
        "**/*research*.py",
        "**/*discovery*.py", 
        "**/*analysis*.py",
        "**/*experiment*.py",
        "**/*hypothesis*.py",
        "**/orchestrator*.py",
        "**/*config*.yml",
        "**/*config*.yaml",
        "**/*config*.json"
    ]
    
    for pattern in patterns:
        research_files.extend(base_path.glob(pattern))
        
    print(f"🔍 Validating {len(research_files)} research files for framework compliance...")
    
    for file_path in research_files:
        if file_path.is_file() and not any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules']):
            result = validator.validate_research_file(file_path)
            results.append(result)
            
            # Print immediate feedback
            status = "✅" if result['compliant'] else "❌"
            print(f"{status} {file_path.name}: {len(result['violations'])} violations, {len(result['warnings'])} warnings")
            
    # Generate and save report
    report = validator.generate_compliance_report(results)
    
    report_path = Path(directory) / ".ironforge" / "research_framework_compliance_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    
    print(f"\n📊 Full compliance report saved: {report_path}")
    
    # Summary
    compliant_count = sum(1 for r in results if r['compliant'])
    total_count = len(results)
    
    if compliant_count == total_count:
        print(f"🎉 All {total_count} files are framework compliant!")
    else:
        print(f"⚠️ {total_count - compliant_count}/{total_count} files need framework improvements")
        print("Run the validator regularly to maintain research framework standards")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate IRONFORGE research framework compliance")
    parser.add_argument("--directory", "-d", default=".", help="Directory to validate")
    parser.add_argument("--strict", "-s", action="store_true", help="Strict validation mode")
    
    args = parser.parse_args()
    
    try:
        validate_ironforge_research(args.directory)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)