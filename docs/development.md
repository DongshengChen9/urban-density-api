# Development And Verification

Supported candidate Python versions are 3.11 and 3.12. Create an isolated
environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[test,docs]"
```

Run the offline public test suite:

```powershell
.venv\Scripts\python -m pytest 03_code/tests -q
```

Validate the documentation:

```powershell
.venv\Scripts\python -m mkdocs build --strict
```

Network acquisition is deliberately excluded from the offline tests. Overture,
OpenStreetMap, and optional height-enrichment access should be tested separately
with a small analysis area and current provider terms.

