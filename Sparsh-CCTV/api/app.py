from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routes_surveillance import router as surveillance_router
from api.routes_settings import router as settings_router
from api.routes_incidents import router as incidents_router
from api.routes_emergency import router as emergency_router
from api.routes_system import router as system_router
from api.dashboard import router as dashboard_router
from surveillance_engine import get_surveillance_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown event handling."""
    print("[FastAPI] Sparsh CCTV Agentic Surveillance API starting up...")
    yield
    print("[FastAPI] Shutting down surveillance engine...")
    engine = get_surveillance_engine(auto_init_detectors=False)
    if engine.is_running:
        engine.stop()


def create_app() -> FastAPI:
    """FastAPI application factory for Sparsh CCTV AI Agentic Surveillance System."""
    app = FastAPI(
        title="Sparsh CCTV Agentic Surveillance & Emergency Response API",
        description=(
            "Real-time video surveillance pipeline, hierarchical accident severity detection, "
            "fire/smoke classification, automated multi-department emergency dispatching, "
            "and dynamic runtime configuration."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(dashboard_router)
    app.include_router(surveillance_router)
    app.include_router(settings_router)
    app.include_router(incidents_router)
    app.include_router(emergency_router)
    app.include_router(system_router)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    return app


app = create_app()
