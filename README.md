# Multi-Platform Trading Bot

🚀 AI-powered trading bot for Binance (Crypto) and Exness (Forex).

## Features

- **Multi-Exchange**: Binance (Crypto) + Exness/MT5 (Forex)
- **Multi-AI**: Gemini, OpenAI, Qwen, Groq, Hugging Face with failover
- **Advanced Strategies**: Trend Following, Mean Reversion, Grid, DCA
- **Backtesting**: Walk-forward analysis, Monte Carlo simulation
- **Risk Management**: Position sizing, daily loss limits, circuit breaker
- **Notifications**: Telegram, Email, Discord
- **Dashboard**: Web UI + REST API

## Quick Start

### Prerequisites

- Python 3.13+
- uv (recommended) or pip

### Installation

```bash
# Clone repository
git clone <repo-url>
cd trading-platform

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Run Paper Trading

```bash
# IMPORTANT: Always start with paper trading!
python -m packages.core.main --paper
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=packages --cov-report=html

# Property-based tests only
pytest -m property
```

## Project Structure

```
trading-platform/
├── packages/
│   ├── core/           # Trading engine, risk manager
│   ├── connectors/     # Exchange connectors (Binance, MT5)
│   ├── ai_analyzer/    # AI providers with failover
│   ├── data_store/     # Database, repositories
│   ├── backtester/     # Backtesting engine
│   ├── strategies/     # Trading strategies
│   ├── quant/          # ML optimizer, portfolio
│   └── shared/         # Common utilities
├── services/
│   ├── api/            # REST API (FastAPI)
│   ├── telegram_bot/   # Telegram notifications
│   └── web_dashboard/  # React dashboard
├── infrastructure/
│   ├── docker/         # Docker configs
│   └── k8s/            # Kubernetes manifests
└── docs/               # Documentation
```

## Safety Rules

⚠️ **IMPORTANT**: This bot trades real money. Follow these rules:

1. **Paper Trading First**: Run 24+ hours in paper mode before live
2. **Start Small**: Begin with minimum position sizes
3. **Monitor**: Check bot status regularly
4. **Risk Limits**: Never disable risk management
5. **Backtest**: Test strategies on historical data first

## License

MIT
