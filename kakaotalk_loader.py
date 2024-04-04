from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.helpers import detect_file_encodings
import pandas as pd
from langchain_community.document_loaders import CSVLoader
import re
import csv
from datetime import datetime


class KaKaoTalkCSVLoader(CSVLoader):
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

    @staticmethod
    def is_relevant_message(message: str) -> bool:
        """메시지가 입장 또는 퇴장 관련 메시지인지 확인합니다."""
        if "님이 들어왔습니다." in message or "님이 나갔습니다." in message:
            return False  # 입장 또는 퇴장 메시지인 경우
        return True  # 그렇지 않은 경우

    def __read_file(self, csvfile) -> Iterator[Document]:
        df = pd.read_csv(csvfile)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date_strf"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").astype(str)
        for i, row in df.iterrows():
            is_relevant = KaKaoTalkCSVLoader.is_relevant_message(row["Message"])
            if not is_relevant:
                continue
            date = row["Date"]
            user = self.anonymize_user_id(row["User"])
            content = f'"User: {user}, Message: {row["Message"]}'

            metadata = {
                "date": row["Date_strf"],
                "year": date.year,
                "month": date.month,
                "day": date.day,
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


class KakaoTalkText2CSVConverter:
    def __init__(self, filepath: str):
        self.filepath = filepath

    @staticmethod
    def convert_datetime(date_str, time_str):
        """날짜와 시간 문자열을 파싱하여 'YYYY-MM-DD HH:MM:SS' 형식의 문자열로 변환합니다."""
        date_formatted = datetime.strptime(date_str.strip(","), "%Y. %m. %d").date()
        if "오후" in time_str:
            hour, minute = map(int, time_str.replace("오후 ", "").split(":"))
            hour = hour + 12 if hour < 12 else hour
        else:
            hour, minute = map(int, time_str.replace("오전 ", "").split(":"))
        new_time = f"{hour:02d}:{minute:02d}:00"
        return f"{date_formatted} {new_time}"

    def process_final_chat_to_csv(self, chat_lines):
        """채팅 로그를 분석하여 CSV 형식으로 변환된 데이터를 리스트로 반환합니다."""
        new_message_pattern = re.compile(
            r"(\d{4}\.\s\d{1,2}\.\s\d{1,2})\.\s(오전|오후)\s(\d{1,2}:\d{2}),\s([^:]+)\s:"
        )
        system_message_pattern = re.compile(
            r"\d{4}년\s\d{1,2}월\s\d{1,2}일\s[월화수목금토일]요일|님이 (들어왔습니다|나갔습니다)\."
        )

        processed_chat_data = []
        current_date, current_time, current_user, current_message = None, None, None, ""

        for line in chat_lines:
            if system_message_pattern.search(line):
                continue

            new_message_match = new_message_pattern.match(line)
            if new_message_match:
                if current_user:
                    current_datetime = self.convert_datetime(
                        current_date, f"{new_message_match.group(2)} {current_time}"
                    )
                    processed_chat_data.append(
                        [
                            current_datetime,
                            current_user,
                            current_message.strip().replace("\n", " "),
                        ]
                    )
                current_date, current_time, current_user = (
                    new_message_match.groups()[0],
                    new_message_match.groups()[2],
                    new_message_match.groups()[3],
                )
                current_message = line[new_message_match.end() :].strip()
            elif line.strip():
                current_message += " " + line.strip()

        if current_user:
            current_datetime = self.convert_datetime(
                current_date, f"{new_message_match.group(2)} {current_time}"
            )
            processed_chat_data.append(
                [
                    current_datetime,
                    current_user,
                    current_message.strip().replace("\n", " "),
                ]
            )

        return processed_chat_data

    def read_chat_file(self, encoding="utf-8"):
        """파일 경로에서 채팅 파일을 읽어오고, 각 줄을 리스트로 반환합니다."""
        try:
            with open(self.filepath, "r", encoding=encoding) as file:
                chat_lines = file.readlines()
            return chat_lines
        except FileNotFoundError:
            print(f"Error: The file {self.filepath} was not found.")
            return []

    def convert(self, encoding="utf-8"):
        """채팅 파일을 읽어서 CSV 파일로 변환하는 주 함수입니다."""
        chat_lines = self.read_chat_file(encoding=encoding)
        if not chat_lines:
            return "Conversion failed due to file read error."

        chat_data_final = self.process_final_chat_to_csv(chat_lines)
        output_path = self.filepath.replace(".txt", ".csv")

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Date", "User", "Message"])
                csvwriter.writerows(chat_data_final)
        except IOError as e:
            print(f"Error writing to CSV: {e}")
            return "Conversion failed due to file write error."

        return output_path
