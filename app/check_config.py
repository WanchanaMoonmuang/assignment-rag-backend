import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

from app.settings import get_settings


async def check() -> int:
    settings = get_settings()
    missing = [
        name
        for name, value in {
            "GEMINI_API_KEY": settings.gemini_api_key,
            "MONGODB_URI": settings.mongodb_uri,
            "MONGODB_DATABASE": settings.mongodb_database,
            "MONGODB_VECTOR_INDEX": settings.mongodb_vector_index,
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

    print("Backend config ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(check()))
