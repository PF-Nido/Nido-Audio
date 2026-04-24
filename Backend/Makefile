# Makefile (en la raíz del proyecto)

COMPOSE = docker compose -f docker/docker-compose.yml

.PHONY: setup up down logs db-shell

setup:
	pip install -e ".[dev]"
	dvc pull

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

db-shell:
	docker exec -it nido-db psql -U nido_user -d nido