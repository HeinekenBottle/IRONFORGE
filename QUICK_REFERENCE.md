# 🚀 Quick Reference Card

## When Stuck - Use This Exact Process:

### 1️⃣ Get Context (30 seconds)
```bash
cd /Users/jack/IRONFORGE
./context_helpers/workflow_helper.sh quick
```

### 2️⃣ Ask Claude the Right Question

**🐛 For Bugs:**
```
"Use the general-purpose agent to search for [your issue] and suggest debugging steps."
```

**🏗️ For Architecture:**  
```
"Use the knowledge-architect agent to explain how [component] should work and identify issues."
```

**📊 For Analysis:**
```
"Use the data-scientist agent to analyze this [data/pattern] and prioritize what's important."
```

**🎯 For General Help:**
```
"I'm stuck on [specific problem]. Break this into steps and tell me exactly what to do next."
```

---

## 🛠️ Essential Commands

```bash
# Quick health check
./context_helpers/workflow_helper.sh quick

# Bug investigation
./context_helpers/workflow_helper.sh bug [search_term]

# Architecture analysis  
./context_helpers/workflow_helper.sh arch [component]

# Save findings for later
./context_helpers/save_reports.sh bug [topic]
./context_helpers/save_reports.sh arch [component]
```

---

## 🆘 Emergency Questions for Claude

### Completely Lost:
```
"I'm working on [goal] but stuck because [obstacle]. Use the general-purpose agent to break this into manageable steps."
```

### Error Messages:
```
"Getting error: [paste error]. Use the general-purpose agent to search IRONFORGE for similar issues and solutions."
```

### Need Ideas:
```
"Need to [goal]. Suggest 3 different approaches with pros/cons using the knowledge-architect agent."
```

### Understanding Code:
```
"Explain how [file/function] works and its role in IRONFORGE using the general-purpose agent."
```

---

## 📍 Quick File Locations

- **This guide**: `/Users/jack/IRONFORGE/WHEN_STUCK.md`
- **Full workflow**: `/Users/jack/IRONFORGE/WORKFLOW_GUIDE.md`  
- **IRONFORGE docs**: `/Users/jack/IRONFORGE/CLAUDE.md`
- **Your reports**: `/Users/jack/IRONFORGE/context_reports/`
- **Context tools**: `/Users/jack/IRONFORGE/context_helpers/`

---

**💡 Remember: Start with context, be specific, ask for steps!**