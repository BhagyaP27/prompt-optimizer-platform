"""
ML Inference Service - FastAPI Application
Serves fine-tuned T5 model for prompt optimization

Architecture:
- FastAPI for async request handling
- PyTorch for model inference
- Prometheus for metrics collection
- Redis for caching (optional)
- Pydantic for request validation

Author: Your Name
Date: 2024
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import logging
from datetime import datetime
import os
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import psutil

# ============================================
# Configuration & Logging
# ============================================

# Configure structured logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
DEVICE = os.getenv("DEVICE", "cpu")

# ============================================
# FastAPI Application Setup
# ============================================

app = FastAPI(
    title="Prompt Optimizer ML Service",
    description="Fine-tuned T5 model for transforming casual prompts into engineered prompts",
    version=MODEL_VERSION,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc alternative
)

# CORS middleware - allows frontend to make requests
# In production, restrict origins to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Prometheus Metrics
# ============================================
# These metrics are scraped by Prometheus for monitoring

# Counter: Monotonically increasing value
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['status', 'model_version']  # Labels for filtering
)

# Histogram: Distribution of values (e.g., latency)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time taken for prediction',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Latency buckets
)

# Gauge: Value that can go up or down
model_load_time = Gauge(
    'model_load_time_seconds',
    'Time taken to load model at startup'
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage'
)

cpu_usage = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

# Mount Prometheus metrics endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ============================================
# Global Model State
# ============================================
# Loaded once at startup, shared across requests

model = None
tokenizer = None
device = None

# ============================================
# Request/Response Models (Pydantic)
# ============================================
# Pydantic automatically validates incoming data

class OptimizeRequest(BaseModel):
    """Request schema for prompt optimization"""
    prompt: str = Field(
        ...,  # Required field
        min_length=1,
        max_length=1000,
        description="User prompt to optimize"
    )
    max_length: Optional[int] = Field(
        512,
        ge=50,  # Greater than or equal to
        le=1024,  # Less than or equal to
        description="Maximum length of generated prompt"
    )
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, higher=more random)"
    )
    num_beams: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Number of beams for beam search"
    )
    top_p: Optional[float] = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Custom validation for prompt content"""
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Security: Check for suspicious patterns
        suspicious = ['<script>', 'javascript:', 'onerror=', 'onclick=']
        for pattern in suspicious:
            if pattern.lower() in v.lower():
                raise ValueError(f'Invalid content detected: {pattern}')
        
        return v
    
    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "prompt": "explain quantum physics",
                "max_length": 512,
                "temperature": 0.7,
                "num_beams": 4
            }
        }

class OptimizeResponse(BaseModel):
    """Response schema for successful optimization"""
    original_prompt: str
    optimized_prompt: str
    model_version: str
    processing_time_ms: float
    confidence: Optional[float] = None

class BatchOptimizeRequest(BaseModel):
    """Request schema for batch optimization"""
    prompts: List[str] = Field(..., min_items=1, max_items=10)
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    model_version: str
    device: str
    uptime_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_available: bool

class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: str
    detail: str
    timestamp: str

# ============================================
# Startup & Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Load model at application startup
    This happens once when container starts
    """
    global model, tokenizer, device
    
    try:
        start_time = time.time()
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Determine device (CPU or GPU)
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer from HuggingFace format
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()  # Disables dropout, batchnorm training mode
        
        # Calculate and log load time
        load_time = time.time() - start_time
        model_load_time.set(load_time)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model parameters: {num_params:,}")
        
        # Warm-up inference (first run is slower due to JIT compilation)
        logger.info("Running warm-up inference...")
        warmup_input = tokenizer(
            "optimize prompt: test",
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(device)
        
        with torch.no_grad():  # Disable gradient computation
            _ = model.generate(**warmup_input, max_length=50)
        
        logger.info("Service ready!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise  # Crash container if model doesn't load

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ML service...")
    # Could add cleanup here (close connections, etc.)

# ============================================
# Middleware
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests and add processing time header
    Middleware runs for every request
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions gracefully"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("NODE_ENV") != "production" else "Internal error",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Prompt Optimizer ML Service",
        "version": MODEL_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "batch": "/optimize/batch",
            "model_info": "/model/info",
            "docs": "/docs",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Used by Docker, Kubernetes, load balancers
    """
    process = psutil.Process()
    
    # Update system metrics
    cpu_percent = psutil.cpu_percent()
    cpu_usage.set(cpu_percent)
    
    # GPU memory if available
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_mem = torch.cuda.memory_allocated()
        gpu_memory_usage.set(gpu_mem)
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        device=str(device),
        uptime_seconds=time.time() - process.create_time(),
        cpu_usage_percent=cpu_percent,
        memory_usage_mb=process.memory_info().rss / 1024 / 1024,
        gpu_available=gpu_available
    )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "T5ForConditionalGeneration",
        "model_version": MODEL_VERSION,
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "max_input_length": 256,
        "max_output_length": 512
    }

@app.post("/optimize", response_model=OptimizeResponse, tags=["Inference"])
async def optimize_prompt(request: OptimizeRequest):
    """
    Optimize a single prompt
    Main inference endpoint
    """
    if model is None or tokenizer is None:
        predictions_total.labels(status='error', model_version=MODEL_VERSION).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Prepare input with task prefix (T5 uses task prefixes)
        input_text = f"optimize prompt: {request.prompt}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate optimized prompt
        with torch.no_grad():  # Don't compute gradients (inference only)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=request.max_length,
                num_beams=request.num_beams,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                top_p=request.top_p,
                early_stopping=True,
                do_sample=request.temperature > 0  # Sampling vs greedy
            )
        
        # Decode output tokens to text
        optimized_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update Prometheus metrics
        predictions_total.labels(status='success', model_version=MODEL_VERSION).inc()
        prediction_latency.observe(processing_time / 1000)
        
        logger.info(f"Optimized prompt in {processing_time:.2f}ms")
        
        return OptimizeResponse(
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        predictions_total.labels(status='error', model_version=MODEL_VERSION).inc()
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/batch", tags=["Inference"])
async def optimize_batch(request: BatchOptimizeRequest):
    """
    Optimize multiple prompts in a single request
    More efficient than individual requests
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Prepare inputs with task prefix
        input_texts = [f"optimize prompt: {p}" for p in request.prompts]
        
        # Tokenize batch
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=request.max_length,
                num_beams=4,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                early_stopping=True
            )
        
        # Decode all outputs
        optimized_prompts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics for batch
        predictions_total.labels(
            status='success',
            model_version=MODEL_VERSION
        ).inc(len(request.prompts))
        
        return {
            "results": [
                {
                    "original": orig,
                    "optimized": opt
                }
                for orig, opt in zip(request.prompts, optimized_prompts)
            ],
            "total_processing_time_ms": processing_time,
            "model_version": MODEL_VERSION
        }
    
    except Exception as e:
        predictions_total.labels(status='error', model_version=MODEL_VERSION).inc()
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )