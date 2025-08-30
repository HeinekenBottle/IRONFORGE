# CURSOR AGENT DOCUMENTATION COMMANDS & VALIDATION

## 🔍 DISCOVERY COMMANDS FOR DOCUMENTATION AUDIT

### Complete Documentation Inventory
```bash
# Find all markdown files (primary documentation)
find . -name "*.md" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/venv/*" | sort

# Find all README files across project
find . -name "README*" -not -path "*/node_modules/*" -not -path "*/.git/*" | sort

# Find other documentation formats
find . \( -name "*.rst" -o -name "*.txt" -o -name "CHANGELOG*" -o -name "CONTRIBUTING*" -o -name "LICENSE*" \) | sort

# Count total documentation files
find . -name "*.md" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/venv/*" | wc -l

# Windows compatibility notes:
# On Windows with Git Bash or WSL, these commands work as-is
# On Windows PowerShell, use:
# Get-ChildItem -Recurse -Include "*.md" | Where-Object { $_.FullName -notlike "*node_modules*" -and $_.FullName -notlike "*\.git\*" }
```

### Code Analysis for Missing Documentation
```bash
# Count Python modules needing documentation
find . -name "*.py" -path "*/ironforge/*" | wc -l

# Find undocumented Python modules
find ironforge/ -name "*.py" -exec basename {} .py \; | sort > /tmp/py_files.txt
find . -name "*.md" -exec grep -l "ironforge\." {} \; | wc -l

# Identify large undocumented directories
find ironforge/ -type d -name "*" | while read dir; do
  if [ ! -f "$dir/README.md" ] && [ $(find "$dir" -name "*.py" | wc -l) -gt 3 ]; then
    echo "Missing README: $dir ($(find "$dir" -name "*.py" | wc -l) Python files)"
  fi
done
```

---

## 🔧 TECHNICAL VALIDATION COMMANDS

### Version Reference Audit
```bash
# Find outdated version references
grep -r "v0\." . --include="*.md" | grep -v node_modules
grep -r "version.*0\." . --include="*.md" | grep -v node_modules
grep -r "1\.0\." . --include="*.md" | grep -v node_modules

# Find references to old CLI commands
grep -r "discover_temporal" . --include="*.md"
grep -r "score_session" . --include="*.md"
grep -r "validate_run" . --include="*.md"
```

### Golden Invariant Verification
```bash
# Verify event type documentation consistency
grep -r "6 types" . --include="*.md" | head -5
grep -r "Expansion.*Consolidation.*Retracement" . --include="*.md" | head -3

# Verify edge intent documentation
grep -r "4 types" . --include="*.md" | head -3
grep -r "TEMPORAL_NEXT.*MOVEMENT_TRANSITION" . --include="*.md" | head -3

# Verify feature dimensions
grep -r "51D.*20D" . --include="*.md" | head -3
grep -r "f45.*f50" . --include="*.md" | head -3
```

### API Documentation Validation  
```bash
# Check for proper API import examples
grep -r "from ironforge.api import" . --include="*.md" | wc -l
grep -r "import ironforge" . --include="*.md" | grep -v "from ironforge.api"

# Verify CLI command documentation
grep -r "discover-temporal" . --include="*.md" | wc -l
grep -r "score-session" . --include="*.md" | wc -l
grep -r "validate-run" . --include="*.md" | wc -l
grep -r "report-minimal" . --include="*.md" | wc -l
```

---

## 📊 CURRENT DOCUMENTATION STATUS (Baseline)

### Existing Documentation Files (As of Analysis)
```
Root Level:
- README.md (main project overview)
- CLAUDE.md (AI assistant guidance)
- QUICK_START.md (getting started)
- QUICK_REFERENCE.md (command reference)

Documentation Directory:
- docs/migrations/ (version migration guides)
- docs/releases/ (release notes and checklists)
- docs/06-TROUBLESHOOTING.md

Agent Systems:
- agents/htf_cascade_predictor/README.md
- agents/pipeline_performance_monitor/README.md  
- agents/archaeological_zone_detector/README.md

Archon Subproject:
- archon/README.md (separate system)
- archon/docs/ (complete documentation system)

Tools & Utilities:
- tools/indexer/README.md
- archive/data_migration/README.md
- tests/_golden/README.md
```

### Documentation Gaps Identified
- **Missing**: Complete API reference for 89 Python modules
- **Missing**: Developer onboarding guide
- **Missing**: Architecture overview document  
- **Missing**: Deployment and production guide
- **Fragmented**: Agent system documentation inconsistent
- **Outdated**: Many version references need updating to v1.1.0

---

## ✅ QUALITY ASSURANCE CHECKLIST

### Pre-Documentation Work
- [ ] Create backup of current documentation state
- [ ] Verify current branch is `chore/agent-planning-docs-and-integration-test-fixes`
- [ ] Confirm IRONFORGE is at v1.1.0 architecture
- [ ] Test current CLI commands to ensure accuracy

### During Documentation Work
- [ ] Maintain Golden Invariants (6 events, 4 edge intents, 51D/20D)
- [ ] Use `from ironforge.api import` pattern in examples
- [ ] Include performance specs (>87% authenticity, <3s sessions)
- [ ] Test all code examples before including
- [ ] Ensure internal links work correctly

### Post-Documentation Validation
```bash
# Test all CLI commands mentioned in docs
python -m ironforge.sdk.cli discover-temporal --help
python -m ironforge.sdk.cli score-session --help  
python -m ironforge.sdk.cli validate-run --help
python -m ironforge.sdk.cli report-minimal --help
python -m ironforge.sdk.cli status --help

# Verify API imports work
python -c "from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash; print('✅ API imports successful')"

# Check that container system is documented correctly
python -c "from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading; print('✅ Container system documented correctly')"

# Verify performance claims
python tools/smoke_checks.py  # Should complete in <3 seconds per session
```

---

## 🎯 SUCCESS METRICS

### Quantitative Targets
- [ ] **100% of Python modules** have at least basic documentation
- [ ] **Zero outdated version references** (all v1.1.0+)  
- [ ] **All CLI commands** have working examples
- [ ] **All code examples** execute without errors
- [ ] **Zero broken internal links** in documentation
- [ ] **Consistent formatting** across all markdown files

### Qualitative Targets  
- [ ] **New developer onboarding** takes <30 minutes with documentation
- [ ] **API usage patterns** are clear from documentation alone
- [ ] **Troubleshooting guide** covers common issues comprehensively
- [ ] **Architecture overview** explains system design clearly
- [ ] **Multi-agent system** integration is well documented

---

## 🚀 EXECUTION WORKFLOW

### Phase 1: Assessment (30 minutes)
1. Run discovery commands to create baseline inventory
2. Identify highest priority missing documentation
3. Verify current technical specifications
4. Test existing code examples for accuracy

### Phase 2: Core Documentation (2-3 hours) 
1. Update root README.md with current v1.1.0 architecture
2. Enhance QUICK_START.md with working examples
3. Standardize QUICK_REFERENCE.md CLI commands
4. Preserve CLAUDE.md structure while updating technical details

### Phase 3: API & Developer Docs (2-3 hours)
1. Create comprehensive docs/API_REFERENCE.md
2. Write docs/DEVELOPER_GUIDE.md for contributors
3. Generate docs/ARCHITECTURE.md system overview
4. Document all major ironforge/* modules

### Phase 4: Agent System Documentation (1-2 hours)
1. Standardize all agents/*/README.md files
2. Document agent integration patterns
3. Create agent usage examples
4. Cross-reference agent capabilities

### Phase 5: Quality Assurance (1 hour)
1. Run all validation commands
2. Test code examples
3. Verify link integrity  
4. Final formatting pass

---

## 📞 TECHNICAL SUPPORT INFORMATION

### Key Technical Facts for Documentation
- **Current Version**: IRONFORGE v1.1.0
- **Python Files**: 89 modules in main ironforge/ package
- **Performance**: <3s per session, <180s full discovery, >87% authenticity
- **Architecture**: Rule-based → TGAT ML → Rule-based pipeline
- **CLI Commands**: discover-temporal, score-session, validate-run, report-minimal
- **API Pattern**: `from ironforge.api import` (recommended)
- **Container**: Lazy loading with dependency injection
- **Quality Gate**: 87% authenticity threshold for production

### Repository Information
- **GitHub**: HeinekenBottle/IRONFORGE
- **Current Branch**: chore/agent-planning-docs-and-integration-test-fixes  
- **Documentation Strategy**: Comprehensive overhaul for production readiness
- **Target Audience**: Developers, researchers, production users

Use these commands and validation steps throughout the documentation overhaul process to ensure technical accuracy and completeness.