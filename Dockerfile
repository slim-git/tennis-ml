# ---- Build stage ----
    FROM python:3.11-slim-bullseye

    COPY requirements.txt .
    COPY requirements-dev.txt .
        
    ARG TEST
    
    # Install dependencies
    RUN pip install --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt && \
        if [ "$TEST" = "true" ]; then \
          pip install --no-cache-dir -r requirements-dev.txt; \
        fi
    
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        curl && \
        rm -rf /var/lib/apt/lists/*
    
    # Copier le code
    COPY ./entrypoint.sh /tmp/entrypoint.sh
    COPY ./src /app/src
    COPY ./tests /app/tests
    COPY ./pytest.ini /app/pytest.ini
    COPY ./.env.test /app/.env.test
    
    WORKDIR /app
    
    # Pour que les imports soient résolus depuis /app
    ENV PYTHONPATH=/app
    
    # Exposer le port
    EXPOSE 7878
    
    # Healthcheck sur FastAPI
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
      CMD curl -f http://localhost:7878/check_health || exit 1
    
    # Utilisateur non-root pour la sécurité
    RUN useradd --create-home appuser
    
    # Give permissions to the appuser
    RUN chown -R appuser:appuser /app
    
    USER appuser
    
    # Entrypoint (par exemple pour lancer uvicorn)
    CMD ["/tmp/entrypoint.sh"]
    