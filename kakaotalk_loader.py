from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.helpers import detect_file_encodings
import pandas as pd
from langchain_community.document_loaders import CSVLoader


class KaKaoTalkLoader(CSVLoader):
    def __init__(self, file_path: str, encoding: str = "utf8", **kwargs):
        super().__init__(file_path, encoding=encoding, **kwargs)

    def anonymize_user_id(self, user_id, num_chars_to_anonymize=3):
        """
        비식별화 함수는 주어진 사용자 ID의 앞부분을 '*'로 대체하여 비식별화합니다.

        :param user_id: 비식별화할 사용자 ID
        :param num_chars_to_anonymize: 비식별화할 문자 수
        :return: 비식별화된 사용자 ID
        """
        # 비식별화할 문자 수가 사용자 ID의 길이보다 길 경우, 전체 ID를 '*'로 대체
        if num_chars_to_anonymize >= len(user_id):
            num_chars_to_anonymize = len(user_id) - 1
            return "*" * num_chars_to_anonymize

        # 앞부분을 '*'로 대체하고 나머지 부분을 원본 ID에서 가져옴
        anonymized_id = "*" * num_chars_to_anonymize + user_id[num_chars_to_anonymize:]

        return anonymized_id

    def __read_file(self, csvfile) -> Iterator[Document]:
        df = pd.read_csv(csvfile)
        for i, row in df.iterrows():
            date = row["Date"]
            user = self.anonymize_user_id(row["User"])
            content = f'"User: {user}, Message: {row["Message"]}'

            metadata = {
                "date": date,
                "user": user,
                "row": i,
                "source": str(self.file_path),
            }
            yield Document(page_content=content, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                yield from self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            yield from self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e
