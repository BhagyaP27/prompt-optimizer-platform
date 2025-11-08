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
