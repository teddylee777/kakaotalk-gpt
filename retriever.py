from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever


class RetrieverFactory(ABC):
    """
    기본 검색기 생성자 클래스입니다. 모든 검색기 팩토리는 이 클래스를 상속받아야 합니다.
    """

    def __init__(self, db: Any):
        """
        검색기 팩토리를 초기화합니다.

        :param db: 데이터베이스 인스턴스
        """
        self.db = db

    @abstractmethod
    def create(self, **kwargs) -> BaseRetriever:
        """
        검색기 인스턴스를 생성하고 반환합니다. 파생 클래스는 이 메서드를 구현해야 합니다.

        :param kwargs: 검색기 생성에 필요한 추가 매개변수
        :return: 생성된 검색기 인스턴스
        """
        pass

    def _configure_fields(
        self,
        retriever: BaseRetriever,
        configurable_fields: Dict[str, ConfigurableField],
    ) -> BaseRetriever:
        """
        검색기에 대한 구성 가능한 필드를 설정합니다.

        :param retriever: 구성할 검색기 인스턴스
        :param configurable_fields: 구성 가능한 필드 딕셔너리
        :return: 구성된 검색기 인스턴스
        """
        for field_name, field_value in configurable_fields.items():
            setattr(retriever, field_name, field_value)
        return retriever


class FAISSRetrieverFactory(RetrieverFactory):
    """
    FAISS 기반 검색기 생성자 클래스입니다.
    """

    def create(self, **kwargs) -> BaseRetriever:
        search_kwargs = kwargs.get("search_kwargs", {"k": 30})
        faiss_retriever = self.db.as_retriever(  # 검색 시 반환할 결과의 개수를 설정합니다.
            search_kwargs=search_kwargs
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                # 검색 매개변수의 고유 식별자를 설정합니다.
                id="search_kwargs_faiss",
                # 검색 매개변수의 이름을 설정합니다.
                name="Search Kwargs",
                # 검색 매개변수에 대한 설명을 작성합니다.
                description="The search kwargs to use",
            )
        )
        return faiss_retriever


class SelfQueryRetrieverFactory(RetrieverFactory):
    """
    SelfQuery 검색기 생성자 클래스입니다.
    """

    def create(self, **kwargs) -> BaseRetriever:
        llm_for_selfquery = ChatOpenAI(
            model=kwargs.get("model", "gpt-4-turbo-preview"),
            temperature=kwargs.get("temperature", 0),
            api_key=kwargs.get("api_key", ""),
        )
        document_content_description = kwargs.get(
            "document_content_description",
            "Information about each chat message in a KakaoTalk chat log.",
        )
        metadata_field_info = kwargs.get(
            "metadata_field_info",
            [
                AttributeInfo(
                    name="year",
                    description="The year of the chat message was sent",
                    type="integer",
                ),
                AttributeInfo(
                    name="month",
                    description="The month of the chat message was sent",
                    type="integer",
                ),
                AttributeInfo(
                    name="day",
                    description="The day of the chat message was sent",
                    type="integer",
                ),
                AttributeInfo(
                    name="user",
                    description="The user who sent the message",
                    type="string",
                ),
                AttributeInfo(
                    name="row",
                    description="The row number in the original CSV file",
                    type="integer",
                ),
                AttributeInfo(
                    name="source", description="File path of the source", type="string"
                ),
            ],
        )

        search_kwargs = kwargs.get("search_kwargs", {"k": 30})
        self_query_retriever = SelfQueryRetriever.from_llm(
            llm_for_selfquery,
            self.db,
            document_content_description,
            metadata_field_info,
            search_kwargs=search_kwargs,
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                # 검색 매개변수의 고유 식별자를 설정합니다.
                id="search_kwargs_selfquery",
                # 검색 매개변수의 이름을 설정합니다.
                name="Search Kwargs",
                # 검색 매개변수에 대한 설명을 작성합니다.
                description="The search kwargs to use",
            )
        )
        return self_query_retriever


class EnsembleRetrieverFactory(RetrieverFactory):
    """
    앙상블 검색기 생성자 클래스입니다.
    """

    def create(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float],
        search_type: str = "mmr",
        **kwargs
    ) -> BaseRetriever:
        if not retrievers or not weights or len(retrievers) != len(weights):
            raise ValueError(
                "Retrievers and weights must be non-empty lists of equal length."
            )

        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
            search_type=search_type,
        )
        return ensemble_retriever
