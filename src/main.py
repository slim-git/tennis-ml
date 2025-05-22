import os
import joblib
import logging
import secrets
from typing import Generator, Optional, Annotated, List
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Query,
    Security,
    Depends
)
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.background import BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from starlette.status import (
    HTTP_200_OK,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_503_SERVICE_UNAVAILABLE)
from dotenv import load_dotenv
from mlflow.exceptions import RestException

from src.entity.model import ModelInput, ModelOutput
from src.service.data_quality import DataChecker, check_model_data
from src.service.model import (
    run_experiment,
    predict,
    list_registered_models,
    load_model,
    deploy_model,
    undeploy_model,
)
from src.repository.common import get_connection
from psycopg import Connection

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def provide_connection() -> Generator[Connection, None, None]:
    with get_connection() as conn:
        yield conn

# ------------------------------------------------------------------------------

FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")
safe_clients = ['127.0.0.1']

api_key_header = APIKeyHeader(name='Authorization', auto_error=False)

async def validate_api_key(request: Request, key: str = Security(api_key_header)):
    '''
    Check if the API key is valid

    Args:
        key (str): The API key to check
    
    Raises:
        HTTPException: If the API key is invalid
    '''
    if request.client.host not in safe_clients and not secrets.compare_digest(str(key), str(FASTAPI_API_KEY)):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Unauthorized - API Key is wrong"
        )
    return None

app = FastAPI(dependencies=[Depends(validate_api_key)] if FASTAPI_API_KEY else None,
              title="Tennis Insights ML API",
              description="API for the Tennis Insights ML module",)

# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    '''
    Redirect to the API documentation.
    '''
    return RedirectResponse(url='/docs')

@app.get("/run_experiment", tags=["model"], description="Schedule a run of the ML experiment")
async def run_xp(background_tasks: BackgroundTasks,
                algo: str = Query(default="LogisticRegression", description="The algorithm to use for training"),
                registered_model_name: Optional[str] = Query(default=None, description="The name of the registered model"),
                experiment_name: Optional[str] = Query(default="Tennis Prediction", description="The name of the experiment")):
    """
    Train the model
    """
    background_tasks.add_task(func=run_experiment,
                              algo=algo,
                              registered_model_name=registered_model_name,
                              experiment_name=experiment_name,)
    
    return {"message": "Experiment scheduled"}

@app.get("/predict",
         tags=["model"],
         description="Predict the outcome of a tennis match",
         response_model=ModelOutput)
async def make_prediction(params: Annotated[ModelInput, Query()]):
    """
    Predict the matches
    """
    if not params.model:
        # check the presence of 'model.pkl' file in data/
        if not os.path.exists("/data/model.pkl"):
            return {"message": "Model not trained. Please train the model first."}
    
        # Load the model
        pipeline = joblib.load("/data/model.pkl")
    else:
        # Get the model info
        try:
            pipeline = load_model(name=params.model, alias=params.alias)
        except RestException as e:
            logger.error(e)

            # Return HTTP error 404
            return HTTPException(
                status=HTTP_404_NOT_FOUND,
                detail=f"Model {params.model} not found"
            )

    # Make the prediction
    prediction = predict(
        pipeline=pipeline,
        series=params.series,
        surface=params.surface,
        court=params.court,
        p1_rank=params.p1_rank,
        p1_play_hand=params.p1_play_hand,
        p1_back_hand=params.p1_back_hand,
        p1_height=params.p1_height,
        p1_weight=params.p1_weight,
        p1_year_of_birth=params.p1_year_of_birth,
        p1_pro_year=params.p1_pro_year,
        p2_rank=params.p2_rank,
        p2_play_hand=params.p2_play_hand,
        p2_back_hand=params.p2_back_hand,
        p2_height=params.p2_height,
        p2_weight=params.p2_weight,
        p2_year_of_birth=params.p2_year_of_birth,
        p2_pro_year=params.p2_pro_year,
    )

    logger.info(prediction)

    return prediction

@app.get("/list_available_models", tags=["model"], description="List the available models")
async def list_available_models(
    aliases: Optional[List[str]] = Query(default=None, description="List of model aliases to filter the models")):
    """
    List the available models
    """
    return list_registered_models(alias_filter=aliases)

@app.post("/deploy_model", tags=["model"], description="Deploy a model")
async def deploy_model_to_production(
    model_name: str = Query(description="The name of the model to deploy"),
    version: str = Query(description="The version of the model to deploy")):
    """
    Deploy a model
    """
    # Deploy the model
    try:
        deploy_model(model_name=model_name, model_version=version)
    except RestException as e:
        logger.error(e)

        # Return HTTP error 404
        return JSONResponse(content={"message": f"Model {model_name} (version {version}) not found"},
                            status_code=HTTP_404_NOT_FOUND)

    return {"message": f"Model {model_name} deployed to production"}

@app.post("/undeploy_model", tags=["model"], description="Undeploy a model")
async def undeploy_model_from_production(model_name: str = Query(description="The name of the model to undeploy")):
    """
    Undeploy a model
    """
    # Undeploy the model
    try:
        undeploy_model(model_name=model_name)
    except RestException as e:
        logger.error(e)

        # Return HTTP error 404
        return JSONResponse(content={"message": f"Model {model_name} not found or not in production"},
                            status_code=HTTP_404_NOT_FOUND)

    return {"message": f"Model {model_name} undeployed from production"}

@app.get("/check_data_quality", tags=["data"], description="Check the data quality")
async def check_data_quality(
    background_tasks: BackgroundTasks,
    model_name: str = Query(description="The name of the model to check"),
    project_id: Optional[str] = Query(default=None, description="The ID of the project to send the data quality report to"),
):
    """
    Check the data quality
    """
    # Get the API key and project ID from the environment variables
    api_key = os.getenv("EVIDENTLY_API_KEY")
    project_id = project_id or os.getenv("EVIDENTLY_PROJECT_ID")

    # Check if the API key and project ID are set
    if not api_key or not project_id:
        return JSONResponse(content={"message": "Evidently API key or project ID not set"},
                            status_code=HTTP_503_SERVICE_UNAVAILABLE)
    
    # Schedule the data quality check
    background_tasks.add_task(func=check_model_data,
                              model_name=model_name,
                              checker=DataChecker(api_key, project_id))
    
    return {"message": "Data quality check scheduled"}

# ------------------------------------------------------------------------------
@app.get("/check_health", tags=["general"], description="Check the health of the ML module")
async def check_health(session: Connection = Depends(provide_connection)):
    """
    Check all the services in the infrastructure are working
    """
    # Check if the database is alive
    try:
        with session.cursor() as cursor:
            cursor.execute("SELECT 1").fetchall()
    except Exception as e:
        logger.error(f"DB check failed: {e}")
        return JSONResponse(content={"status": "unhealthy", "detail": "Database not reachable"},
                            status_code=HTTP_503_SERVICE_UNAVAILABLE)
    
    # Check if the mlflow endpoint is reachable
    if MLFLOW_SERVER_URI := os.getenv("MLFLOW_SERVER_URI"):
        import requests

        try:
            # Ping the mlflow server endpoint
            response = requests.get(MLFLOW_SERVER_URI + "/health", timeout=5)
            if response.status_code != HTTP_200_OK:
                logger.error(f"Mlfow server check failed: {response.status_code}")
                return JSONResponse(content={"status": "unhealthy", "detail": "Mlfow server not reachable"},
                                    status_code=HTTP_503_SERVICE_UNAVAILABLE)
        except requests.RequestException as e:
            logger.error(f"Mlfow server check failed: {e}")
            return JSONResponse(content={"status": "unhealthy", "detail": "Mlfow server not reachable"},
                                status_code=HTTP_503_SERVICE_UNAVAILABLE)
    
    return JSONResponse(content={"status": "healthy"}, status_code=HTTP_200_OK)
