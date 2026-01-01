"""
Configuration management using pydantic-settings.

Loads from environment variables and .env files.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingConfig(BaseSettings):
    """Trading-specific configuration."""

    paper_mode: bool = Field(default=True, description="Paper trading mode (no real orders)")
    default_timeframe: str = Field(default="1h", description="Default candlestick timeframe")
    max_position_pct: Decimal = Field(
        default=Decimal("0.05"), description="Max position size as % of balance"
    )
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.03"), description="Max daily loss as % of balance"
    )
    max_open_positions: int = Field(default=5, description="Max concurrent open positions")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Min AI confidence to trade"
    )

    @field_validator("max_position_pct", "max_daily_loss_pct", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal) -> Decimal:
        """Parse string/float to Decimal."""
        return Decimal(str(v))


class BinanceConfig(BaseSettings):
    """Binance exchange configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: SecretStr = Field(default=SecretStr(""), description="Binance API key")
    api_secret: SecretStr = Field(default=SecretStr(""), description="Binance API secret")
    testnet: bool = Field(default=True, description="Use Binance testnet")
    symbols: list[str] = Field(default=["BTCUSDT", "ETHUSDT"], description="Trading symbols")

    @property
    def is_configured(self) -> bool:
        """Check if Binance credentials are configured."""
        return bool(self.api_key.get_secret_value() and self.api_secret.get_secret_value())


class ExnessConfig(BaseSettings):
    """Exness/MT5 configuration."""

    model_config = SettingsConfigDict(env_prefix="MT5_")

    login: int = Field(default=0, description="MT5 account login")
    password: SecretStr = Field(default=SecretStr(""), description="MT5 password")
    server: str = Field(default="Exness-MT5Trial", description="MT5 server")
    symbols: list[str] = Field(default=["EURUSD", "XAUUSD"], description="Trading symbols")

    @property
    def is_configured(self) -> bool:
        """Check if MT5 credentials are configured."""
        return bool(self.login and self.password.get_secret_value())


class AIConfig(BaseSettings):
    """AI provider configuration."""

    primary_provider: str = Field(default="gemini", description="Primary AI provider")
    fallback_chain: list[str] = Field(
        default=["openai", "groq"], description="Fallback provider chain"
    )
    cache_ttl_seconds: int = Field(default=300, description="AI response cache TTL")
    timeout_seconds: int = Field(default=30, description="AI request timeout")

    # API Keys
    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    groq_api_key: SecretStr = Field(default=SecretStr(""), alias="GROQ_API_KEY")
    qwen_api_key: SecretStr = Field(default=SecretStr(""), alias="QWEN_API_KEY")
    huggingface_api_key: SecretStr = Field(default=SecretStr(""), alias="HUGGINGFACE_API_KEY")


class TelegramConfig(BaseSettings):
    """Telegram notification configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")

    bot_token: SecretStr = Field(default=SecretStr(""), description="Telegram bot token")
    chat_id: str = Field(default="", description="Telegram chat ID")
    enabled: bool = Field(default=True, description="Enable Telegram notifications")

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is configured."""
        return bool(self.bot_token.get_secret_value() and self.chat_id)


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(default="sqlite:///data/trading.db", alias="DATABASE_URL")
    echo: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(default=5, description="Connection pool size")


class RedisConfig(BaseSettings):
    """Redis configuration."""

    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    enabled: bool = Field(default=False, description="Enable Redis caching")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App settings
    app_name: str = Field(default="Trading Platform", description="Application name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Sub-configs
    trading: TradingConfig = Field(default_factory=TradingConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    exness: ExnessConfig = Field(default_factory=ExnessConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def validate_for_live_trading(self) -> list[str]:
        """Validate configuration for live trading. Returns list of errors."""
        errors = []

        if self.trading.paper_mode:
            errors.append("Paper mode is enabled - disable for live trading")

        if not self.binance.is_configured and not self.exness.is_configured:
            errors.append("No exchange credentials configured")

        if self.binance.testnet and self.binance.is_configured:
            errors.append("Binance testnet is enabled - disable for live trading")

        return errors


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
