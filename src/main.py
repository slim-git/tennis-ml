import os
import joblib
import logging
import secrets
from typing import Generator, Literal, Optional, Annotated
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
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mlflow.exceptions import RestException

from src.service.model import (
    run_experiment,
    predict,
    list_registered_models,
    load_model,
    all_algorithms,
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
                 algo: all_algorithms = 'LogisticRegression',
                 registered_model_name: Optional[str] = None,
                 experiment_name: str = 'Tennis Prediction'):
    """
    Train the model
    """
    background_tasks.add_task(func=run_experiment,
                              algo=algo,
                              registered_model_name=registered_model_name,
                              experiment_name=experiment_name,)
    
    return {"message": "Experiment scheduled"}

class ModelInput(BaseModel):
    p1_rank: int = Field(gt=0, default=1, description="The rank of the 1st player")
    p2_rank: int = Field(gt=0, default=100, description="The rank of the 2nd player")
    court: Literal['Outdoor', 'Indoor'] = 'Outdoor'
    surface: Literal['Grass', 'Carpet', 'Clay', 'Hard'] = 'Clay'
    series: Literal['Grand Slam', 'Masters 1000', 'Masters', 'Masters Cup', 'ATP500', 'ATP250', 'International Gold', 'International'] = 'Grand Slam'
    p1_height: Optional[int] = Field(gt=0, default=180, description="The height of the 1st player in centimeters")
    p2_height: Optional[int] = Field(gt=0, default=180, description="The height of the 2nd player in centimeters")
    p1_weight: Optional[int] = Field(gt=0, default=80, description="The weight of the 1st player in kilograms")
    p2_weight: Optional[int] = Field(gt=0, default=80, description="The weight of the 2nd player in kilograms")
    p1_year_of_birth: Optional[int] = Field(gt=1950, default=1980, description="The year of birth of the 1st player")
    p2_year_of_birth: Optional[int] = Field(gt=1950, default=1980, description="The year of birth of the 2nd player")
    p1_play_hand: Literal['Right', 'Left'] = 'Right'
    p2_play_hand: Literal['Right', 'Left'] = 'Right'
    p1_back_hand: Literal[1, 2] = 1
    p2_back_hand: Literal[1, 2] = 1
    p1_pro_year: Optional[int] = Field(gt=1970, default=2000, description="The year the 1st player turned pro")
    p2_pro_year: Optional[int] = Field(gt=1970, default=2000, description="The year the 2nd player turned pro")
    model: Optional[str] = 'LogisticRegression'
    version: Optional[str] = 'latest'

class ModelOutput(BaseModel):
    result: int = Field(description="The prediction result. 1 if player 1 is expected to win, 0 otherwise.", json_schema_extra={"example": "1"})
    prob: list[float] = Field(description="Probability of [defeat, victory] of player 1.", json_schema_extra={"example": "[0.15, 0.85]"})

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
            pipeline = load_model(params.model, params.version)
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
async def list_available_models():
    """
    List the available models
    """
    return list_registered_models()

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
