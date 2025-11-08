"""
Pydantic schemas for request/response validation
Exports all schemas for easy importing
"""

from .request import OptimizeRequest, BatchOptimizeRequest
from .response import OptimizeResponse, BatchOptimizeResponse, HealthResponse, ErrorResponse

__all__ = [
    'OptimizeRequest',
    'BatchOptimizeRequest',
    'OptimizeResponse',
    'BatchOptimizeResponse',
    'HealthResponse',
    'ErrorResponse'
]


# ============================================
# ml-service/schemas/request.py
# ============================================
"""
Request schemas for API endpoints
Pydantic models validate incoming data automatically
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List


class OptimizeRequest(BaseModel):
    """
    Schema for single prompt optimization request
    
    Validates:
    - Prompt is not empty and within length limits
    - Parameters are within valid ranges
    - No malicious content in prompt
    """
    
    prompt: str = Field(
        ...,  # Required field (ellipsis means required)
        min_length=1,
        max_length=1000,
        description="User prompt to optimize",
        example="explain quantum physics"
    )
    
    max_length: Optional[int] = Field(
        512,
        ge=50,  # Greater than or equal
        le=1024,  # Less than or equal
        description="Maximum length of generated prompt"
    )
    
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0=deterministic, higher=more random"
    )
    
    num_beams: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Number of beams for beam search (higher=better quality, slower)"
    )
    
    top_p: Optional[float] = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (alternative to temperature)"
    )
    
    use_cache: Optional[bool] = Field(
        True,
        description="Whether to use cached results if available"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """
        Custom validation for prompt content
        
        Steps:
        1. Clean whitespace
        2. Check for malicious patterns
        3. Ensure reasonable length
        """
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Security: Check for XSS patterns
        suspicious_patterns = [
            '<script>', 'javascript:', 'onerror=', 'onclick=',
            'onload=', '<iframe>', 'eval(', 'document.cookie'
        ]
        
        for pattern in suspicious_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f'Invalid content detected: {pattern}')
        
        # Check if prompt is just whitespace
        if not v.strip():
            raise ValueError('Prompt cannot be empty or just whitespace')
        
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v, values):
        """
        Temperature should be 0 for deterministic output
        or > 0 for sampling
        """
        if v == 0:
            # Deterministic mode
            return v
        elif v < 0.1:
            raise ValueError('Temperature should be 0 (deterministic) or >= 0.1')
        return v
    
    class Config:
        """Pydantic configuration"""
        schema_extra = {
            "example": {
                "prompt": "explain machine learning to a 5 year old",
                "max_length": 512,
                "temperature": 0.7,
                "num_beams": 4,
                "top_p": 0.9,
                "use_cache": True
            }
        }


class BatchOptimizeRequest(BaseModel):
    """
    Schema for batch prompt optimization
    
    Allows optimizing multiple prompts in one request
    More efficient than multiple single requests
    """
    
    prompts: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,  # Limit to prevent overload
        description="List of prompts to optimize"
    )
    
    max_length: Optional[int] = Field(
        512,
        ge=50,
        le=1024,
        description="Maximum length for all generated prompts"
    )
    
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    
    num_beams: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Number of beams for beam search"
    )
    
    @validator('prompts')
    def validate_prompts(cls, v):
        """Validate each prompt in the batch"""
        cleaned_prompts = []
        
        for prompt in v:
            # Clean whitespace
            cleaned = ' '.join(prompt.split())
            
            # Check length
            if len(cleaned) < 1 or len(cleaned) > 1000:
                raise ValueError(f'Prompt length must be between 1 and 1000 characters')
            
            # Check for malicious content
            suspicious = ['<script>', 'javascript:', 'onerror=']
            for pattern in suspicious:
                if pattern.lower() in cleaned.lower():
                    raise ValueError(f'Invalid content in prompt: {pattern}')
            
            cleaned_prompts.append(cleaned)
        
        return cleaned_prompts
    
    class Config:
        schema_extra = {
            "example": {
                "prompts": [
                    "explain AI",
                    "write a poem about nature",
                    "help me code a sorting algorithm"
                ],
                "max_length": 512,
                "temperature": 0.7,
                "num_beams": 4
            }
        }


# ============================================
# ml-service/schemas/response.py
# ============================================
"""
Response schemas for API endpoints
Defines structure of successful responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class OptimizeResponse(BaseModel):
    """
    Schema for successful single prompt optimization
    
    Returns both original and optimized prompts
    plus metadata about the generation
    """
    
    original_prompt: str = Field(
        ...,
        description="Original user prompt"
    )
    
    optimized_prompt: str = Field(
        ...,
        description="Optimized engineered prompt"
    )
    
    model_version: str = Field(
        ...,
        description="Version of model used"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken for inference in milliseconds"
    )
    
    confidence: Optional[float] = Field(
        None,
        description="Model confidence score (if available)",
        ge=0.0,
        le=1.0
    )
    
    cached: Optional[bool] = Field(
        False,
        description="Whether result was returned from cache"
    )
    
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of generation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "original_prompt": "explain AI",
                "optimized_prompt": "Act as an AI expert. Provide a comprehensive explanation...",
                "model_version": "v1.0.0",
                "processing_time_ms": 1250.5,
                "confidence": 0.92,
                "cached": False,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchOptimizeItem(BaseModel):
    """Single item in batch response"""
    original: str
    optimized: str
    processing_time_ms: Optional[float] = None


class BatchOptimizeResponse(BaseModel):
    """
    Schema for batch optimization response
    Contains array of results plus aggregate stats
    """
    
    results: List[BatchOptimizeItem] = Field(
        ...,
        description="Array of optimization results"
    )
    
    total_processing_time_ms: float = Field(
        ...,
        description="Total time for batch processing"
    )
    
    model_version: str = Field(
        ...,
        description="Model version used"
    )
    
    count: int = Field(
        ...,
        description="Number of prompts processed"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "original": "explain AI",
                        "optimized": "Act as an AI expert...",
                        "processing_time_ms": 1200
                    },
                    {
                        "original": "write a poem",
                        "optimized": "You are a creative poet...",
                        "processing_time_ms": 1300
                    }
                ],
                "total_processing_time_ms": 2500,
                "model_version": "v1.0.0",
                "count": 2
            }
        }


class HealthResponse(BaseModel):
    """
    Schema for health check endpoint
    Used by Docker, Kubernetes, load balancers
    """
    
    status: str = Field(
        ...,
        description="Health status: 'healthy' or 'unhealthy'"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded and ready"
    )
    
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    
    device: str = Field(
        ...,
        description="Device being used (cpu/cuda)"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )
    
    cpu_usage_percent: float = Field(
        ...,
        description="Current CPU usage percentage"
    )
    
    memory_usage_mb: float = Field(
        ...,
        description="Current memory usage in MB"
    )
    
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available"
    )
    
    gpu_memory_used_mb: Optional[float] = Field(
        None,
        description="GPU memory used in MB (if GPU available)"
    )


class ErrorResponse(BaseModel):
    """
    Schema for error responses
    Consistent error format across all endpoints
    """
    
    error: str = Field(
        ...,
        description="Error type or category"
    )
    
    detail: str = Field(
        ...,
        description="Detailed error message"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp when error occurred"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracking"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Prompt exceeds maximum length of 1000 characters",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_abc123"
            }
        }


# ============================================
# ml-service/schemas/common.py
# ============================================
"""
Common schemas used across multiple endpoints
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class ModelInfo(BaseModel):
    """Model metadata"""
    model_type: str
    model_version: str
    framework: str
    framework_version: str
    device: str
    parameters: int
    max_input_length: int
    max_output_length: int


class MetricsResponse(BaseModel):
    """Metrics snapshot"""
    predictions_total: int
    predictions_success: int
    predictions_error: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    uptime_seconds: float


class StatusResponse(BaseModel):
    """General status information"""
    service: str
    version: str
    status: str
    endpoints: Dict[str, str]