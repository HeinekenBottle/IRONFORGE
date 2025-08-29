# 🆘 When You're Stuck - Emergency Guide

**When you don't know what to do, follow this step-by-step process.**

## 🚀 Quick Start: The 3-Question Method

Ask yourself these 3 questions in order:

### 1. "What am I trying to accomplish?"
- **Bug fixing?** → Go to [Bug Investigation](#-bug-investigation)
- **Understanding code?** → Go to [Code Understanding](#-code-understanding)  
- **Making architecture decisions?** → Go to [Architecture Decisions](#-architecture-decisions)
- **Performance issues?** → Go to [Performance Investigation](#-performance-investigation)
- **Something else?** → Go to [General Problem Solving](#-general-problem-solving)

### 2. "What context do I have?"
- **None** → Start with [Context Gathering](#-context-gathering)
- **Some files/errors** → Use [Targeted Analysis](#-targeted-analysis)
- **Lots of information** → Use [Information Organization](#-information-organization)

### 3. "What's my next action?"
- Use the specific workflows below based on your answers to #1 and #2

---

## 🐛 Bug Investigation

### When You're Stuck:
```bash
# Quick overview of potential issues
cd /Users/jack/IRONFORGE
./context_helpers/workflow_helper.sh quick
```

### What to Ask Claude:
```
"I'm seeing [specific error/behavior]. Use the general-purpose agent to search the codebase for similar issues and common causes."
```

### If That Doesn't Help:
```bash
# Deep bug analysis
./context_helpers/workflow_helper.sh bug [search_term]
```

Then ask Claude:
```
"Based on this bug hunting report, use the general-purpose agent to prioritize which files to investigate first for [your specific issue]."
```

---

## 🧠 Code Understanding  

### When You Don't Understand Code:
```
"Use the general-purpose agent to explain how [specific file/function/class] works and its role in the IRONFORGE system."
```

### For Complex Systems:
```bash
# Get architecture overview
./context_helpers/workflow_helper.sh arch [component_name]
```

Then ask Claude:
```
"Use the knowledge-architect agent to explain this architecture analysis in simple terms and show me the key relationships."
```

### For Specific Functions:
```
"Walk me through this code step by step: [paste code snippet]. What does each part do and why?"
```

---

## 🏗️ Architecture Decisions

### When You Need to Make Design Choices:
```bash
# Analyze current architecture
./context_helpers/workflow_helper.sh arch [component]
```

### What to Ask Claude:
```
"Based on this architecture analysis, use the knowledge-architect agent to suggest 3 design approaches for [your specific goal] and explain the tradeoffs."
```

### For Major Changes:
```
"I need to [describe your goal]. Use the knowledge-architect agent to analyze the impact on the existing IRONFORGE architecture and suggest an implementation plan."
```

---

## ⚡ Performance Investigation

### When Things Are Slow:
```
"Use the data-scientist agent to analyze the performance characteristics of [specific component/function] and identify bottlenecks."
```

### For System-Wide Issues:
```bash
# Check for performance indicators
./context_helpers/workflow_helper.sh quick
```

Then ask Claude:
```
"Based on these performance indicators, use the data-scientist agent to prioritize optimization opportunities and estimate impact."
```

---

## 🎯 General Problem Solving

### When You're Completely Lost:
```
"I'm working on [brief description] and I'm stuck because [what's blocking you]. Use the general-purpose agent to break this down into smaller, manageable steps."
```

### When You Have an Error Message:
```
"I'm getting this error: [paste error]. Use the general-purpose agent to search the IRONFORGE codebase for similar errors and their solutions."
```

### When You Need Ideas:
```
"I need to [describe goal] but I'm not sure of the best approach. Use the general-purpose agent to suggest 3 different strategies with pros and cons."
```

---

## 📋 Context Gathering

### Start Here When You Have No Context:
```bash
# Quick system overview
cd /Users/jack/IRONFORGE
./context_helpers/workflow_helper.sh quick
```

### Get Specific Context:
```bash
# For bugs
./context_helpers/save_reports.sh bug [search_term]

# For architecture
./context_helpers/save_reports.sh arch [component]
```

### Build Progressive Context:
1. Start broad: `./context_helpers/workflow_helper.sh quick`
2. Get specific: `./context_helpers/workflow_helper.sh bug [topic]`
3. Analyze: Feed results to Claude Code agents
4. Document: Save findings to build knowledge

---

## 🎯 Targeted Analysis

### When You Have Specific Files/Errors:
```
"Use the general-purpose agent to analyze these specific files: [list files] and explain how they relate to [your problem]."
```

### When You Have Error Messages:
```
"I'm getting this error in [file:line]: [error message]. Use the general-purpose agent to search for related code patterns and suggest fixes."
```

---

## 📚 Information Organization

### When You Have Too Much Information:
```
"I have all this information [paste/describe]. Use the knowledge-architect agent to organize this into key themes and prioritize what to address first."
```

### When You Need to Make Sense of Findings:
```
"Based on these analysis results [paste results], use the data-scientist agent to identify the most important insights and recommend next actions."
```

---

## 🚨 Emergency Cheat Sheet

**Copy and paste these exact commands when stuck:**

### 🔍 Quick Diagnosis:
```bash
cd /Users/jack/IRONFORGE && ./context_helpers/workflow_helper.sh quick
```

### 🐛 Bug Investigation:
```
"Use the general-purpose agent to search the IRONFORGE codebase for issues related to [your problem] and suggest debugging steps."
```

### 🏗️ Architecture Help:
```
"Use the knowledge-architect agent to explain how [component/feature] should work in the IRONFORGE system and identify potential issues."
```

### 📊 Data Analysis:
```
"Use the data-scientist agent to analyze this information [paste data/logs] and tell me what's most important to focus on."
```

### 🎯 Next Steps:
```
"I've analyzed [what you did] and found [brief summary]. What should I do next to [your goal]?"
```

---

## 💡 Pro Tips for Getting Unstuck

1. **Start Small**: Don't try to understand everything at once
2. **Be Specific**: Instead of "this doesn't work", say "function X returns Y but I expected Z"
3. **Use Context Tools**: Always run context gathering before asking complex questions
4. **Save Your Work**: Use `save_reports.sh` to document findings
5. **Ask for Alternatives**: "What are 3 different ways to approach this?"
6. **Break It Down**: "Help me break this big problem into smaller pieces"

---

## 📍 Where to Look for Information

### In Your System:
- **Recent findings**: `/Users/jack/IRONFORGE/context_reports/`
- **Workflow guide**: `/Users/jack/IRONFORGE/WORKFLOW_GUIDE.md`
- **This guide**: `/Users/jack/IRONFORGE/WHEN_STUCK.md`

### In IRONFORGE:
- **Main documentation**: `/Users/jack/IRONFORGE/CLAUDE.md`
- **Pipeline info**: `/docs/DATA_PIPELINE_ARCHITECTURE.md`
- **Recent changes**: `git log --oneline -10`

### In Archon:
- **Knowledge base**: Running on `http://localhost:8181`
- **Agent patterns**: `/Users/jack/IRONFORGE/archon/documentation_assistant/`

---

## 🎪 The "I'm Really Stuck" Nuclear Option

When nothing else works, ask Claude this:

```
"I'm completely stuck on [brief problem description]. I've tried [what you attempted]. Please:

1. Use the general-purpose agent to search the IRONFORGE codebase for relevant patterns
2. Suggest 3 completely different approaches to this problem  
3. Tell me which approach to try first and exactly what commands to run
4. Give me the specific Claude Code questions to ask if that doesn't work

My goal is: [be very specific about end goal]
My current state: [describe exactly where you are]
What's blocking me: [specific obstacle]"
```

**This will get you unstuck 95% of the time.**

Remember: Being stuck is normal. Having a systematic approach to getting unstuck is what makes the difference!