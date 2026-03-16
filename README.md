# BIRD Text-to-SQL with LLM

An intelligent Text-to-SQL system using Large Language Models for the [BIRD benchmark](https://bird-bench.github.io/). This project achieves **72.47% execution accuracy** on the BIRD development set using GPT-5.2 with question-type routing and value grounding.

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Dev Set Accuracy** | 72.47% (1066/1471) |
| Model | GPT-5.2 |
| Cost per run | ~$24 |
| Time | ~6 minutes |

### Per-Database Performance

| Database | Accuracy |
|----------|----------|
| superhero | 89.1% |
| student_club | 86.1% |
| european_football_2 | 75.6% |
| codebase_community | 73.7% |
| california_schools | 73.0% |
| toxicology | 71.0% |
| debit_card_specializing | 71.9% |
| thrombosis_prediction | 68.1% |
| formula_1 | 65.5% |
| financial | 65.1% |
| card_games | 62.4% |

## 🚀 Key Features

1. **Question-Type Routing**: Classifies questions (ratio, aggregation, superlative, etc.) and applies type-specific SQL rules
2. **Value Grounding**: Extracts entities from questions and grounds them to actual database values
3. **Database-Specific Context**: Curated examples and rules for each BIRD database
4. **Multi-Candidate Voting**: Generates multiple candidates at different temperatures and votes on execution results
5. **Automatic SQL Repair**: Attempts to fix syntax errors automatically

## 📋 Requirements

- Python 3.10+
- [Modal](https://modal.com/) account (for serverless GPU execution)
- OpenAI API key with GPT-5.2 access

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/chirantangupta05/bird-text-to-sql.git
cd bird-text-to-sql

# Install dependencies
pip install -r requirements.txt

# Set up Modal
modal setup

# Add your OpenAI API key to Modal secrets
modal secret create openai-secret OPENAI_API_KEY=your-key-here
```

## 📦 Data Setup

1. Download the BIRD benchmark from [BIRD-bench](https://bird-bench.github.io/)
2. Create a Modal volume and upload the data:

```bash
modal volume create bird-dataset
# Upload dev.json and databases/ folder to the volume
```

## 🏃 Running

```bash
# Run on full dev set
modal run modal_app.py

# Run on specific database
modal run modal_app.py --db california_schools

# Run on limited questions (for testing)
modal run modal_app.py --limit 50
```

## 📊 Architecture

```
Question + Evidence
        ↓
┌───────────────────┐
│ Question Routing  │ → Classify: ratio/aggregation/superlative/etc.
└───────────────────┘
        ↓
┌───────────────────┐
│ Value Grounding   │ → Extract entities, match to DB values
└───────────────────┘
        ↓
┌───────────────────┐
│ Schema + Context  │ → Database-specific rules & examples
└───────────────────┘
        ↓
┌───────────────────┐
│ Multi-Candidate   │ → Generate 3 SQLs at temp 0.0, 0.3, 0.5
│ Generation        │
└───────────────────┘
        ↓
┌───────────────────┐
│ Execution Voting  │ → Execute all, vote on matching results
└───────────────────┘
        ↓
┌───────────────────┐
│ SQL Repair        │ → Fix syntax errors if needed
└───────────────────┘
        ↓
    Final SQL
```

## 📁 Project Structure

```
bird-text-to-sql/
├── modal_app.py              # Main application
├── requirements.txt          # Python dependencies
├── data/
│   └── descriptions/         # Database context files
│       ├── california_schools.json
│       ├── card_games.json
│       └── ...
└── results/
    └── results_v51.json      # Evaluation results
```

## 🔬 Technical Details

### Question-Type Routing

The system classifies questions into types and applies specific rules:

- **Ratio**: Use `CAST(... AS REAL)` for division
- **Aggregation**: `COUNT(*)`, `SUM()`, `AVG()` patterns
- **Superlative**: `ORDER BY ... LIMIT 1` patterns
- **Date**: Date format conversion rules

### Value Grounding

Extracts potential values from questions and evidence, then searches the database for exact matches. This prevents case sensitivity and spelling errors.

### Database-Specific Rules

Each database has curated context including:
- Column mappings (e.g., `Charter School (Y/N)` uses 1/0 not 'Y'/'N')
- Value formats (e.g., Czech vocabulary in financial database)
- Join patterns
- Few-shot examples

## 📚 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{gupta2026texttosql,
  author = {Chirantan Gupta},
  title = {Intelligent Text-to-SQL System using Large Language Models},
  school = {BITS Pilani},
  year = {2026},
  type = {M.Tech Dissertation}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- [BIRD Benchmark](https://bird-bench.github.io/) team for the dataset
- Prof. Y.V.K. Ravi Kumar (Academic Supervisor)
- Srikanth Valluri, PwC (Industry Supervisor)
