import os.path
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from typing import List
from app.utils import (
    Showcase2dirModel, Showcase4dirModel
)

app = FastAPI()

sc2 = Showcase2dirModel()
sc4 = Showcase4dirModel()


class Datum(BaseModel):
    context: List[str] = [
        "Alice walks north. ",
        "Bob walks north. ",
        "Alice follows Bob. ",
    ]
    actor1: str = "Alice"
    actor2: str = "Bob"


static_path = Path(os.path.dirname(__file__)) / "static"
app.mount(
    "/static",
    StaticFiles(directory=static_path),
    name="static"
)


@app.get("/")
def index():
    with open(Path(os.path.dirname(__file__)) / 'static/index.html') as f:
        data = f.read()
    return HTMLResponse(content=data, status_code=200)


@app.post("/eval/{dirs}")
async def eval_model(dirs: int, datum: Datum):
    if dirs == 2:
        model = sc2
    elif dirs == 4:
        model = sc4
    else:
        raise ValueError("Unkown number of directions")

    try:
        resp = model(datum.context, datum.actor1, datum.actor2)
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(detail=e.__repr__(), status_code=500)
