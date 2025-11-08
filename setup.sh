#!/bin/bash
# Prompt Optimizer Platform - Setup Script
# Run: bash setup.sh

set -e

echo "🚀 Setting up Prompt Optimizer Platform..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Project root
PROJECT_ROOT="prompt-optimizer-platform"

# Create root
mkdir -p $PROJECT_ROOT
cd $PROJECT_ROOT

echo -e "${BLUE}Creating project structure...${NC}"

# Root files
touch README.md .gitignore docker-compose.yml Makefile

# Directories
mkdir -p .github/workflows docs scripts monitoring/grafana/dashboards

# ML Service
mkdir -p ml-service/{api/v1/endpoints,core,models,schemas,utils,tests}
cd ml-service
touch Dockerfile requirements.txt .env.example app.py config.py
touch api/__init__.py api/v1/__init__.py api/v1/endpoints/__init__.py
touch core/__init__.py models/__init__.py schemas/__init__.py utils/__init__.py
touch tests/__init__.py
cd ..

# Backend
mkdir -p backend/{config,routes/v1,controllers,models,middleware,services,utils,tests}
cd backend
touch Dockerfile package.json .env.example server.js
touch config/database.js routes/index.js routes/v1/prompts.js
touch controllers/promptController.js models/Prompt.js
touch middleware/errorHandler.js services/mlService.js utils/logger.js
cd ..

# Frontend
mkdir -p frontend/{public,src/{components/{common,prompt},pages,hooks,services,styles}}
cd frontend
touch Dockerfile package.json .env.example vite.config.js index.html
cd src
touch main.jsx App.jsx
touch components/common/Button.jsx components/prompt/PromptInput.jsx
touch pages/Home.jsx hooks/useOptimizePrompt.js services/api.js
cd ../..

# ML Training
mkdir -p ml-model-training/{data/{raw,processed},config,src/{data,models,training},scripts}
cd ml-model-training
touch requirements.txt README.md
touch config/model_config.yaml
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch scripts/train.py scripts/generate_dataset.py
cd ..

# Models
mkdir -p models/{best_model,checkpoints}

# Infrastructure
mkdir -p infrastructure/{terraform,kubernetes/base}

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
venv/
env/
*.egg-info/

# Node
node_modules/
npm-debug.log*
dist/
build/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Logs
logs/
*.log

# Models
*.pth
*.bin
models/checkpoints/

# Data
data/raw/*.csv
data/processed/*.csv

# Testing
.pytest_cache/
.coverage
EOF

# Create README
cat > README.md << 'EOF'
# Prompt Optimizer Platform

Transform casual prompts into professional, engineered prompts using AI.

## Quick Start

```bash
# Start all services
docker-compose up

# Access:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:5000
# - ML Service: http://localhost:8000
```

## Structure

- `ml-service/` - FastAPI ML inference service
- `backend/` - Express.js API gateway
- `frontend/` - React user interface
- `ml-model-training/` - Model training pipeline
- `models/` - Saved model files

## Development

See individual service READMEs for setup instructions.
EOF

# Create Makefile
cat > Makefile << 'EOF'
.PHONY: help start stop test

help:
	@echo "make start  - Start all services"
	@echo "make stop   - Stop all services"
	@echo "make test   - Run tests"

start:
	docker-compose up

stop:
	docker-compose down

test:
	cd ml-service && pytest
	cd backend && npm test
	cd frontend && npm test
EOF

echo ""
echo -e "${GREEN}✅ Project structure created!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. cd $PROJECT_ROOT"
echo "2. Copy your code files into the appropriate folders"
echo "3. Run: docker-compose up"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"