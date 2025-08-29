# IRONFORGE Documentation Standards

## 📋 Naming Convention

### Core Documentation (docs/)
- **01-QUICKSTART.md** - Getting started guide (installation, first run)
- **02-USER-GUIDE.md** - Complete user guide (daily workflows, best practices)
- **03-API-REFERENCE.md** - API documentation with examples
- **04-ARCHITECTURE.md** - System architecture and design
- **05-DEPLOYMENT.md** - Production deployment and monitoring
- **06-TROUBLESHOOTING.md** - Common issues and solutions
- **07-CHANGELOG.md** - Version history and breaking changes
- **08-GLOSSARY.md** - Terminology and definitions

### Release Documentation (docs/releases/)
- **vX.Y.Z.md** - Release notes for version X.Y.Z
- **templates/** - Release and changelog templates

### Specialized Documentation (docs/specialized/)
- **TGAT-ARCHITECTURE.md** - Technical deep-dives
- **SEMANTIC-FEATURES.md** - Feature documentation
- **PATTERN-DISCOVERY.md** - Advanced guides
- **MCP-INTEGRATION.md** - Context7 MCP guide

### Archive Documentation (docs/archive/)
- **YYYY-MM-DD-description.md** - Historical documentation
- **legacy/** - Deprecated documentation

## 📝 Content Standards

### File Structure
```markdown
# Title - Brief Description
**Version**: X.Y.Z
**Last Updated**: YYYY-MM-DD

## 🎯 Overview
Brief description of the document's purpose

## 📋 Table of Contents
- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1
Content with clear headings and examples

## 🔗 Related Documentation
- [Related Doc 1](link)
- [Related Doc 2](link)
```

### Writing Guidelines
1. **Clear Headings**: Use emoji prefixes for visual scanning
2. **Code Examples**: Include working code snippets
3. **Cross-References**: Link to related documentation
4. **Version Info**: Include version and last updated date
5. **Consistent Formatting**: Use standard markdown conventions

## 🗂️ File Organization Rules

### What Goes Where
- **Core docs/**: Essential user-facing documentation
- **Specialized docs/specialized/**: Technical deep-dives
- **Release docs/releases/**: Version-specific information
- **Archive docs/archive/**: Historical documentation

### What Gets Archived
- Refactor summaries older than 6 months
- Completed project reports
- Outdated migration guides
- Superseded documentation

### What Gets Deleted
- Duplicate files with identical content
- Temporary files (drafts, notes)
- Superseded documentation without historical value
- Empty or placeholder files

## 🔄 Maintenance Process

### Monthly Review
1. Check for outdated documentation
2. Archive completed project reports
3. Update cross-references
4. Validate all links

### Release Process
1. Update changelog
2. Create release notes
3. Update version numbers
4. Archive previous version docs

### New Documentation
1. Follow naming convention
2. Use standard template
3. Include cross-references
4. Add to main index