"""
Configuration Management for ML Service
Centralizes all configuration from environment variables

Why this file:
- Single source of truth for configuration
- Type-safe settings with validation
- Easy to test and modify
- Separates config from business logic
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    Pydantic automatically validates types and provides defaults
    """
    
    # ============================================
    # Model Configuration
    # ============================================
    model_path: str = Field(
        default="/models/best_model",
        description="Path to trained model directory"
    )
    
    model_version: str = Field(
        default="v1.0.0",
        description="Model version for tracking"
    )
    
    device: str = Field(
        default="cpu",
        description="Device for inference: 'cpu' or 'cuda'"
    )
    
    @validator('device')
    def validate_device(cls, v):
        """Ensure device is valid"""
        if v not in ['cpu', 'cuda', 'mps']:  # mps for Apple Silicon
            raise ValueError(f"Invalid device: {v}. Must be 'cpu', 'cuda', or 'mps'")
        return v
    
    # ============================================
    # Server Configuration
    # ============================================
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind server"
    )
    
    port: int = Field(
        default=8000,
        description="Port to bind server"
    )
    
    workers: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # ============================================
    # Inference Configuration
    # ============================================
    max_batch_size: int = Field(
        default=32,
        description="Maximum batch size for inference"
    )
    
    max_input_length: int = Field(
        default=256,
        description="Maximum input sequence length"
    )
    
    max_output_length: int = Field(
        default=512,
        description="Maximum output sequence length"
    )
    
    default_num_beams: int = Field(
        default=4,
        description="Default number of beams for beam search"
    )
    
    default_temperature: float = Field(
        default=0.7,
        description="Default sampling temperature"
    )
    
    # ============================================
    # Caching Configuration
    # ============================================
    enable_cache: bool = Field(
        default=True,
        description="Enable Redis caching"
    )
    
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL"
    )
    
    cache_ttl: int = Field(
        default=3600,
        description="Cache time-to-live in seconds"
    )
    
    # ============================================
    # Logging Configuration
    # ============================================
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Ensure log level is valid"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v
    
    log_format: str = Field(
        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        description="Log message format"
    )
    
    # ============================================
    # Performance Configuration
    # ============================================
    enable_amp: bool = Field(
        default=False,
        description="Enable Automatic Mixed Precision for faster inference"
    )
    
    compile_model: bool = Field(
        default=False,
        description="Enable model compilation (PyTorch 2.0+)"
    )
    
    # ============================================
    # Security Configuration
    # ============================================
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum request size in bytes"
    )
    
    allowed_origins: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # ============================================
    # Monitoring Configuration
    # ============================================
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: int = Field(
        default=8000,
        description="Port for metrics endpoint"
    )
    
    class Config:
        """Pydantic configuration"""
        # Allow environment variables with this prefix
        env_prefix = ""
        # Case sensitive environment variable names
        case_sensitive = False
        # Load from .env file if present
        env_file = ".env"
        env_file_encoding = 'utf-8'


# ============================================
# Global Settings Instance
# ============================================
# Create single instance to be imported throughout app
settings = Settings()


# ============================================
# Helper Functions
# ============================================

def get_settings() -> Settings:
    """
    Get settings instance
    Useful for dependency injection in FastAPI
    """
    return settings


def print_settings():
    """Print current settings (for debugging)"""
    print("=" * 50)
    print("ML Service Configuration")
    print("=" * 50)
    print(f"Model Path:        {settings.model_path}")
    print(f"Model Version:     {settings.model_version}")
    print(f"Device:            {settings.device}")
    print(f"Host:              {settings.host}:{settings.port}")
    print(f"Log Level:         {settings.log_level}")
    print(f"Cache Enabled:     {settings.enable_cache}")
    print(f"Metrics Enabled:   {settings.enable_metrics}")
    print("=" * 50)


# ============================================
# Usage Example
# ============================================
"""
# In other files, import settings:
from config import settings

# Use settings:
model_path = settings.model_path
device = settings.device

# Or use as FastAPI dependency:
from fastapi import Depends
from config import get_settings, Settings

@app.get("/config")
def get_config(settings: Settings = Depends(get_settings)):
    return {"model_version": settings.model_version}
"""