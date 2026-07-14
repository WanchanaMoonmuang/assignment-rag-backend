import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

from app.settings import Settings, get_settings


def provider_missing_settings(settings: Settings) -> list[str]:
    provider = settings.gemini_provider.lower()
    if provider == "developer_api":
        return [] if settings.gemini_api_key else ["GEMINI_API_KEY"]
    if provider == "vertex_ai":
        missing = []
        if not (settings.google_cloud_project or settings.gcp_project_id):
            missing.append("GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID")
        if not settings.google_cloud_location:
            missing.append("GOOGLE_CLOUD_LOCATION")
        return missing
    return ["GEMINI_PROVIDER must be developer_api or vertex_ai"]


async def check() -> int:
    settings = get_settings()
    missing = provider_missing_settings(settings) + [
        name
        for name, value in {
            "MONGODB_URI": settings.mongodb_uri,
            "MONGODB_DATABASE": settings.mongodb_database,
            "MONGODB_VECTOR_INDEX": settings.mongodb_vector_index,
            "MONGODB_SEARCH_INDEX": settings.mongodb_search_index,
            "JWT_SECRET_KEY": settings.jwt_secret_key,
            "AUTH_PASSWORD": settings.auth_password,
        }.items()
        if not value
    ]
    if missing:
        print(f"Missing required settings: {', '.join(missing)}")
        return 1

    client = AsyncIOMotorClient(settings.mongodb_uri)
    try:
        await client.admin.command("ping")
        collection = client[settings.mongodb_database][settings.mongodb_chunk_collection]
        indexes = await collection.aggregate([{"$listSearchIndexes": {}}]).to_list(length=None)
    finally:
        client.close()

    if not any(index.get("name") == settings.mongodb_vector_index for index in indexes):
        print(
            f"Vector index {settings.mongodb_vector_index!r} not found on "
            f"{settings.mongodb_chunk_collection!r}"
        )
        return 1

    if not any(index.get("name") == settings.mongodb_search_index for index in indexes):
        print(
            f"Search index {settings.mongodb_search_index!r} not found on "
            f"{settings.mongodb_chunk_collection!r}"
        )
        return 1

    print("Backend config ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(check()))
