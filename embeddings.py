from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings


def embedding_factory(api_key: str, use_cache=True):
    # OpenAI 임베딩을 사용하여 기본 임베딩 설정
    embedding = OpenAIEmbeddings(api_key=api_key)

    if use_cache:
        # 로컬 파일 저장소 설정
        store = LocalFileStore("./cache/")

        # 캐시를 지원하는 임베딩 생성
        faiss_cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            embedding,
            store,
            namespace="faiss",  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
        )

        chroma_cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            embedding,
            store,
            namespace="chroma",  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
        )

        return {"faiss": faiss_cached_embedder, "chroma": chroma_cached_embedder}
    else:
        return {"faiss": embedding, "chroma": embedding}
