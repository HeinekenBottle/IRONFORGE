# Contract Compliance Enforcer - Dependencies

## Environment Variables

```bash
APP_ENV=development
LOG_LEVEL=INFO
IRONFORGE_CONFIG_PATH=configs/dev.yml
PIPELINE_TIMEOUT_SECONDS=180
AUTHENTICITY_THRESHOLD=0.87
AGENT_COMMUNICATION_TIMEOUT=30
```

## Dependencies

- Python Packages

```txt
jsonschema>=4.23.0
pydantic>=2.8.0
numpy>=1.24.0
```

## Notes

- Environment Variables section present
- Dependencies section present
- Designed for fast validation (<2s) with minimal footprint
