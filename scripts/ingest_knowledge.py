from app.services.knowledge_service import knowledge_service


if __name__ == "__main__":
    count = knowledge_service.ingest()
    print(f"Ingested {count} knowledge chunks.")
