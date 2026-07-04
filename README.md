# EntropyOS

An AI middleware library that maximizes useful information per token. EntropyOS sits between a user/application and an LLM, optimizing prompts by scoring information density, removing filler, deduplicating content, retrieving relevant context from hierarchical memory, and compiling optimized prompts. The goal is to reduce token consumption (and thus cost) while preserving critical information.

## Features

- **Information Scorer**: Heuristic-based and LLM-based scoring (entropy, novelty, redundancy, importance, dependency)
- **Hierarchical Memory**: 5-level memory store (L0-L4) with TTL-based promotion/demotion
- **Semantic Compressor**: Removes filler phrases, hedges, boilerplate while preserving protected content
- **Retrieval Engine**: Vector similarity search with MMR reranking and multiple backends
- **Context Optimizer**: Deduplication, compression, conflict detection, and priority ranking
- **Response Evaluator**: Quality metrics for accuracy, completeness, and hallucination risk
- **Plugin System**: Extensible plugin architecture for custom scoring, compression, and retrieval
- **Security Layer**: PII detection, memory encryption, and audit logging
- **LangChain Integration**: Seamless integration with LangChain applications

## Installation

```bash
pip install entropyos          # Core only
pip install entropyos[api]     # + FastAPI/uvicorn
pip install entropyos[all]     # Full install with all extras
```

## Quick Start

```python
from entropyos import EntropyPipeline

# Initialize pipeline with configuration
config = EntropyConfig()
pipeline = EntropyPipeline(config)

# Process a user prompt
user_prompt = "it is worth noting that the user requested information about machine learning"
optimized_prompt, metrics = pipeline.process(user_prompt)

print(f"Original: {user_prompt}")
print(f"Optimized: {optimized_prompt}")
print(f"Metrics: {metrics}")
```

## Core Components

### 1. EntropyPipeline (`pipeline.py`)
The orchestrator that runs the complete optimization pipeline:
1. PII redaction
2. Scoring
3. Retrieval
4. Prompt compilation
5. Context optimization
6. Evaluation
7. Memory insertion
8. Audit logging

### 2. InformationScorer (`scorer.py`)
Two scoring modes:
- **Heuristic**: Shannon entropy + redundancy + importance + dependency + novelty
- **LLM**: Heuristic + LLM override for complexity metrics

Formula: `info_score = novelty*0.3 + importance*0.35 + dependency*0.2 - redundancy*0.15`

### 3. HierarchicalMemory (`memory.py`)
5 memory tiers with TTL-based promotion/demotion:
- L0 (current, 5min TTL)
- L1 (conversation, 1h)
- L2 (session, 24h)
- L3 (long-term, 7d)
- L4 (archive, no TTL)

### 4. SemanticCompressor (`compressor.py`)
Removes filler phrases, hedges, and instruction boilerplate. Protected content (numbers, names, code, emails, dates) is never removed.

### 5. RetrievalEngine (`retrieval.py`)
Vector similarity search with MMR reranking. Supports:
- N-gram vector backend (built-in)
- Sentence-transformer embedding backend

## Architecture

```
User/App → [EntropyOS Pipeline] → Optimized Prompt → LLM (OpenAI/Anthropic)
              ↓
         [HierarchicalMemory] ←→ [RetrievalEngine]
              ↓
         [AuditLogger] [PIIDetector] [MemoryEncryptor]
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python ≥3.11 |
| API Framework | FastAPI (optional) |
| ASGI Server | uvicorn (optional) |
| LLM Clients | openai, anthropic (optional) |
| Embeddings | sentence-transformers (optional) |
| Caching | redis (optional) |
| Serialization | pickle (file), YAML (config) |
| Testing | pytest |
| Linting | ruff (line-length=100) |
| Encryption | cryptography.fernet (optional) |

## Configuration

### Default Configuration
Create `entropyos.yaml` in your project root:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  timeout_seconds: 30.0

compression_level: 0.5
max_token_budget: 4096
recall_threshold: 0.3

memory:
  backend: local
  l0_ttl_seconds: 300

retrieval:
  engine: vector
  top_k: 5
  use_mmr: true

security:
  encrypt_memory: false
  pii_detection: true
  audit_log: true
```

### Environment Variables
- `ENTROPYOS_LLM_API_KEY`: LLM API key
- `ENTROPYOS_ENCRYPTION_KEY`: Fernet encryption key (base64)
- `ENTROPYOS_CONFIG_PATH`: Path to config file

## Deployment

### Docker
```bash
# Build and run
 docker build -t entropyos . && docker run -p 8000:8000 entropyos

# With docker-compose
 docker-compose up -d
```

### CLI
```bash
# Serve the API
serve --host 0.0.0.0 --port 8000

# Test the pipeline locally
python demo/validate.py
```

## Usage Examples

### Local Pipeline
```python
from entropyos import EntropyPipeline, EntropyConfig

config = EntropyConfig()
pipeline = EntropyPipeline(config)

# Process multiple prompts
prompts = [
    "it is worth noting that the user requested information",
    "could you provide details about the requested information",
    "the information you requested includes machine learning details"
]

for prompt in prompts:
    optimized, metrics = pipeline.process(prompt)
    print(f"Reduced by {metrics['compression_ratio']*100:.1f}%")
```

### API Usage
```bash
# Call the REST API
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "example prompt"}'
```

### LangChain Integration
```python
from langchain.llms import OpenAI
from entropyos.integrations.langchain import EntropyLangchain

llm = EntropyLangchain(OpenAI(temperature=0))
response = llm.invoke("What is machine learning?")
```

## Testing

Run the test suite:

```bash
pytest
```

Test components:
- `tests/test_unit.py`: Unit tests (33 tests)
- `tests/test_integration.py`: Integration tests
- `tests/test_integrations.py`: LangChain integration tests
- `tests/test_benchmark.py`: Performance benchmark tests
- `tests/test_smoke.py`: Quick smoke test
- `tests/test_demo.py`: Demo agent tests

## Examples

Run demo applications:

```bash
# Real-world benchmarks
python demo/benchmark.py

# Validation suite
python demo/validate.py

# Demo agent
python demo/agent.py
```

## Development

### Code Style

- Python 3.11+ with `from __future__ import annotations`
- Dataclasses for models and config
- Abstract base classes for extensibility
- `ruff` with line-length 100
- No docstrings or comments

### Running Tests

```bash
pytest tests/ -v
```

### Adding a Plugin

Plugins can extend EntropyOS functionality. See `plugins/` directory for examples.

## Roadmap

- [ ] Redis memory backend implementation
- [ ] User isolation and access control
- [ ] Streaming support for real-time processing
- [ ] Advanced contradiction detection
- [ ] Automatic rate limiting
- [ ] CORS configuration
- [ ] Production-ready authentication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow existing code conventions
4. Add tests for new functionality
5. Run `pytest` to verify changes

## License

MIT