from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from exceptions import AppException, InvalidPlayerException
from pydantic import BaseModel
from requestmodels import PostGameRequest, PostGamePlayerPlayRequest
from gamemanager import GameManager, players
from sessionmemory import SessionMemory
from auth import verify_credentials
from card import CARDS
import uvicorn

SessionMemory.init()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000", "http://193.168.144.145:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Gestionnaire d'exception
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )

# Filet de sécurité pour les exceptions non prévues
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

# POST /game : débuter une nouvelle partie
# GET /game/{game_uid} : obtenir l'état d'un jeu en cours
# POST /game/{game_uid}/player/play : le joueur joue
# POST /game/{game_uid}/opponent/play : la machine joue
# GET /players : obtenir les joueurs disponibles
# GET /cards : obtenir la liste des cartes


@app.post("/game")
def post_game(request: PostGameRequest, username: str = Depends(verify_credentials)):
    game = GameManager.create(player=request.player)

    return game.get_game_representation()

@app.get("/game/{game_uid}")
def get_game(game_uid: str, username: str = Depends(verify_credentials)):
    game = GameManager(uid=game_uid)

    return game.get_game_representation()

@app.post("/game/{game_uid}/player/play")
def post_game_player_play(request:PostGamePlayerPlayRequest, game_uid:str, username: str = Depends(verify_credentials)):
    game = GameManager(uid=game_uid)
    r = game.player_play(from_pos=(request.from_pos_col, request.from_pos_row), to_pos=(request.to_pos_col, request.to_pos_row), card_idx=request.card_idx)

    return game.get_game_representation()

@app.post("/game/{game_uid}/opponent/play")
def post_game_opponent_play(game_uid:str, username: str = Depends(verify_credentials)):
    game = GameManager(uid=game_uid)
    r = game.opponent_play()

    return game.get_game_representation()

@app.get("/players")
def get_players(username: str = Depends(verify_credentials)):
    out = {
        'players' : []
    }
    for player in players:
        out['players'].append(
            {
                'uid' : player,
                'name' : players[player]['name'],
                'description' : players[player]['description'],
                'stars' : players[player]['stars'],
                'class' : players[player]['class'].name
            }
        )

    return out

@app.get("/cards")
def get_cards(username: str = Depends(verify_credentials)):
    return [
        {k: v for k, v in card.__dict__.items() if k not in ("print_value", "opponent_print_value")}
        for card in CARDS
    ]

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)