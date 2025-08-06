# MCP Implementation and Hosting Guide

This guide covers deploying and hosting MCP servers and clients in production environments using public cloud resources and services.

## Deployment Strategies

### 1. Local Development Setup

```bash
# Development environment setup
git clone https://github.com/your-repo/mcp-implementation.git
cd mcp-implementation

# Virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python server.py  # For stdio transport
uvicorn server:app --host 0.0.0.0 --port 8000  # For HTTP transport
```

### 2. Docker Containerization

#### Dockerfile for MCP Server

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mcpdb
      POSTGRES_USER: mcpuser
      POSTGRES_PASSWORD: mcppass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

## Cloud Hosting Options

### 3. AWS Deployment

#### AWS Lambda Serverless Function

```python
# lambda_handler.py
import json
import asyncio
from mcp import McpServer
from mcp.server.lambda_adapter import lambda_handler_adapter

server = McpServer("lambda-mcp-server")

@server.tool()
async def process_data(data: str) -> str:
    """Process data in AWS Lambda."""
    # Your processing logic here
    return f"Processed: {data}"

# Create Lambda handler
lambda_handler = lambda_handler_adapter(server)

# For testing locally
if __name__ == "__main__":
    # Test event
    test_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "process_data", "arguments": {"data": "test"}},
            "id": 1
        })
    }
    
    result = lambda_handler(test_event, {})
    print(json.dumps(result, indent=2))
```

#### AWS SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    Runtime: python3.11
    Environment:
      Variables:
        LOG_LEVEL: INFO

Resources:
  McpServerFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_handler.lambda_handler
      Events:
        McpApi:
          Type: Api
          Properties:
            Path: /mcp
            Method: post
      Environment:
        Variables:
          ENVIRONMENT: production

  McpApiGateway:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Cors:
        AllowMethods: "'GET,POST,OPTIONS'"
        AllowHeaders: "'content-type'"
        AllowOrigin: "'*'"

Outputs:
  McpApi:
    Description: "API Gateway endpoint URL for MCP server"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/mcp/"
```

#### ECS Fargate Deployment

```yaml
# ecs-task-definition.json
{
  "family": "mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "mcp-server",
      "image": "your-account.dkr.ecr.region.amazonaws.com/mcp-server:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PORT",
          "value": "8000"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mcp-server",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 4. Google Cloud Platform

#### Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/mcp-server', '.']
  
  # Push the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/mcp-server']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'mcp-server'
      - '--image'
      - 'gcr.io/$PROJECT_ID/mcp-server'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--port'
      - '8000'
      - '--memory'
      - '512Mi'
      - '--cpu'
      - '1'
      - '--max-instances'
      - '10'

options:
  logging: CLOUD_LOGGING_ONLY
```

#### Cloud Functions

```python
# main.py for Cloud Functions
import functions_framework
import asyncio
from mcp import McpServer
from mcp.server.cloud_functions_adapter import cloud_functions_adapter

server = McpServer("gcp-mcp-server")

@server.tool()
async def gcp_process_data(data: str) -> str:
    """Process data in Google Cloud Functions."""
    return f"GCP Processed: {data}"

@functions_framework.http
def mcp_handler(request):
    """HTTP Cloud Function entry point."""
    return cloud_functions_adapter(server)(request)
```

### 5. Microsoft Azure

#### Azure Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2018-10-01
location: eastus
name: mcp-server-aci
properties:
  containers:
  - name: mcp-server
    properties:
      image: your-registry.azurecr.io/mcp-server:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 1
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: PORT
        value: '8000'
      - name: ENVIRONMENT
        value: 'production'
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: mcp-server-unique
tags:
  environment: production
  service: mcp-server
type: Microsoft.ContainerInstance/containerGroups
```

#### Azure Functions

```python
# function_app.py
import azure.functions as func
import asyncio
from mcp import McpServer
from mcp.server.azure_functions_adapter import azure_functions_adapter

app = func.FunctionApp()
server = McpServer("azure-mcp-server")

@server.tool()
async def azure_process_data(data: str) -> str:
    """Process data in Azure Functions."""
    return f"Azure Processed: {data}"

@app.route(route="mcp", auth_level=func.AuthLevel.ANONYMOUS)
def mcp_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Functions HTTP trigger."""
    return azure_functions_adapter(server)(req)
```

## Production Configuration

### 6. Environment Configuration

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductionConfig:
    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "4"))
    
    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Authentication
    api_key: str = os.getenv("API_KEY", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    
    # Monitoring
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    
    # Feature flags
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "false").lower() == "true"
    
    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

config = ProductionConfig()
```

### 7. Production Server Setup

```python
# production_server.py
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from mcp import McpServer
from mcp.server.fastapi import create_mcp_router
from config import config

# Configure Sentry for error tracking
if config.sentry_dsn:
    sentry_sdk.init(
        dsn=config.sentry_dsn,
        integrations=[FastApiIntegration(auto_enable=True)],
        traces_sample_rate=0.1,
    )

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mcp_server.log')
    ]
)

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'status'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'MCP request duration')

# Create MCP server
mcp_server = McpServer("production-mcp-server")

@mcp_server.tool()
async def production_tool(data: str) -> str:
    """Production-ready tool with proper error handling."""
    try:
        # Your tool logic here
        logger.info(f"Processing data: {data}")
        result = f"Processed: {data}"
        REQUEST_COUNT.labels(method='production_tool', status='success').inc()
        return result
    except Exception as e:
        logger.error(f"Error in production_tool: {e}")
        REQUEST_COUNT.labels(method='production_tool', status='error').inc()
        raise

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MCP server...")
    yield
    # Shutdown
    logger.info("Shutting down MCP server...")

app = FastAPI(
    title="Production MCP Server",
    description="Production-ready MCP server with monitoring and observability",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return generate_latest()

# Include MCP router
mcp_router = create_mcp_router(mcp_server)
app.include_router(mcp_router, prefix="/mcp")

if __name__ == "__main__":
    uvicorn.run(
        "production_server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower(),
        access_log=True
    )
```

## Monitoring and Observability

### 8. Monitoring Setup

```python
# monitoring.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Metrics
mcp_requests_total = Counter('mcp_requests_total', 'Total requests', ['method', 'status'])
mcp_request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')
mcp_active_connections = Gauge('mcp_active_connections', 'Active connections')

# Structured logging
logger = structlog.get_logger()

def monitor_mcp_call(func):
    """Decorator to monitor MCP tool calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            mcp_requests_total.labels(method=method_name, status='success').inc()
            logger.info("MCP call successful", method=method_name, duration=time.time() - start_time)
            return result
        except Exception as e:
            mcp_requests_total.labels(method=method_name, status='error').inc()
            logger.error("MCP call failed", method=method_name, error=str(e), duration=time.time() - start_time)
            raise
        finally:
            mcp_request_duration.observe(time.time() - start_time)
    
    return wrapper

# Usage in server
@mcp_server.tool()
@monitor_mcp_call
async def monitored_tool(data: str) -> str:
    """Tool with monitoring."""
    return f"Processed: {data}"
```

### 9. Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-server-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - mcp.yourdomain.com
    secretName: mcp-server-tls
  rules:
  - host: mcp.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcp-server-service
            port:
              number: 80
```

## Security and Authentication

### 10. Security Implementation

```python
# security.py
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt

security = HTTPBearer()

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None

auth_manager = AuthManager(config.jwt_secret)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return payload

# Protected MCP endpoint
@app.post("/mcp/protected")
async def protected_mcp_endpoint(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Protected MCP endpoint requiring authentication."""
    # Process MCP request with user context
    return await mcp_server.handle_request(request, user_context=current_user)
```

## Load Balancing and Scaling

### 11. Auto-scaling Configuration

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
```

## CI/CD Pipeline

### 12. GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy MCP Server

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    
    - name: Run tests
      run: pytest tests/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: mcp-server
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster mcp-cluster --service mcp-server --force-new-deployment
```

## Cost Optimization

### 13. Cost-Effective Hosting Strategies

```python
# cost_optimization.py
import asyncio
from datetime import datetime, time

class CostOptimizedServer:
    """MCP server with cost optimization features."""
    
    def __init__(self):
        self.active_hours = (time(8, 0), time(18, 0))  # 8 AM to 6 PM
        self.weekend_mode = False
    
    async def should_scale_down(self) -> bool:
        """Determine if server should scale down for cost savings."""
        now = datetime.now()
        current_time = now.time()
        is_weekend = now.weekday() >= 5
        
        # Scale down outside business hours or on weekends
        if is_weekend or not (self.active_hours[0] <= current_time <= self.active_hours[1]):
            return True
        
        return False
    
    async def optimize_resources(self):
        """Optimize resource usage based on time and demand."""
        if await self.should_scale_down():
            # Reduce worker processes, connection pools, etc.
            await self.scale_down_resources()
        else:
            await self.scale_up_resources()
    
    async def scale_down_resources(self):
        """Scale down resources for cost savings."""
        # Implement resource scaling logic
        pass
    
    async def scale_up_resources(self):
        """Scale up resources for peak performance."""
        # Implement resource scaling logic
        pass
```

## Next Steps

- [Best Practices and Known Issues](./best-practices.md)
- [Complete Implementation Examples](../../patterns/mcp/)

## Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Google Cloud Architecture Center](https://cloud.google.com/architecture)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)