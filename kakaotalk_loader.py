import re
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.helpers import detect_file_encodings
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from datetime import datetime


class KaKaoTalkLoader(CSVLoader):
    def __init__(self, file_path: str, file_suffix:str, encoding: str = "utf8", **kwargs):
        super().__init__(file_path, encoding=encoding, **kwargs)
        # NOTE - choh(2024.04.05) - 파일 확장자 변수 추가
        self.file_suffix = file_suffix
    
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
    
    # NOTE - choh(2024.04.05) - 12시간제를 24시간제로 변환
    def process_time_to_24hr_format(self, date_obj, time_str):
        """
        대화 내용중에 시간 표시가 '오전 12:23', '오후 11:23'과 같이 12시간제로 되어 있는 경우, 
        이를 24시간제로 변환합니다.
        
        :param date_obj: 대화 내용의 날짜 정보가 담긴 datetime 객체
        :praam time_str: 대화 내용의 시간 정보가 담긴 문자열
        :return: 24시간제로 변환된 datetime 객체
        """
        
        # '오전/오후' 부분과 시간 부분을 분리합니다.
        period, time_part = time_str.split(' ', 1)
        
        # 시간 부분을 시와 분으로 다시 분리합니다.
        hour, minute = map(int, time_part.split(':'))
        
        # '오후'인 경우 12를 더하되, '오후 12시'는 제외합니다.
        if period == '오후' and hour != 12:
            hour += 12
        # '오전 12시'는 0시로 처리합니다.
        elif period == '오전' and hour == 12:
            hour = 0
        
        # date_obj과 결합하여 최종 datetime 객체를 생성합니다.
        # 여기서 datetime 함수는 위에서 임포트한 datetime 클래스를 사용합니다.
        combined_datetime = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute)
        
        # pandas의 to_datetime 함수를 사용하여 pandas.Timestamp 객체로 변환합니다.
        return pd.to_datetime(combined_datetime)
    
    # NOTE - choh(2024.04.05) - 대화목록의 날짜 변환 부분을 파싱
    def process_date(self, line: str) -> tuple:
        """
        -------- 2024년 4월 5일 화요일 -------- 형태의 날짜를 파싱하고,
        파싱 성공 여부와 함께 파싱된 날짜 또는 원래 문자열을 반환합니다.
        
        :param line: 날짜 문자열
        :return: (파싱 성공 여부, 파싱된 날짜 또는 원래 문자열)
        """
        # -------- 2024년 4월 5일 화요일 -------- 날짜가 이상태임
        date_match = re.match(r'[-]+ (\d+년 \d+월 \d+일) [^\d]+', line)
        if date_match:
            # 2024년 4월 5일, 형태의 날짜 추출
            date_pattern = re.compile(r'(\d+)년 (\d+)월 (\d+)일')
            match = date_pattern.search(date_match.group(1))
            if match:
                year, month, day = map(int, match.groups())
                return (True, pd.to_datetime(f"{year}-{month}-{day}"))
        return (False, line)

    # NOTE - choh(2024.04.05) - __read_file을 테스트 하기 위한 wrapper 함수
    def _read_file_test(self, csvfile) -> Iterator[Document]:
        """테스트를 위한 래퍼 함수"""
        return self.__read_file(csvfile)
    
    def __read_file(self, csvfile) -> Iterator[Document]:
        # NOTE - choh(2024.04.05) - TXT 형태의 대화 메세지 사전 처리
        if self.file_suffix == ".txt":
            
            # 전날 날짜 변수 초기화
            temp_date = None
            i = 0 # 행 번호
            for line in csvfile:
                
                # 이번 줄이 날짜가 맞으면 is_parsed=True, result는 날짜
                is_parsed, result = self.process_date(line)
                
                # 파싱한 문자열이 날짜 패턴에 맞으면, 날짜를 저장
                if is_parsed:
                    temp_date = result
                
                # 날짜가 아니면, 체팅이기 때문에, 체팅을 패턴 매칭
                else:
                    # 초기값 설정
                    user = None
                    time_12hr = None
                    message = None

                    # 대화 패턴 찾기
                    conversation_match = re.match(r'\[([^\]]+)\] \[([^\]]+)\] (.+)', line)
                    if conversation_match:
                        user_real = conversation_match.group(1)
                        time_12hr = conversation_match.group(2)
                        message = conversation_match.group(3).strip()
                        
                        # 시간을 24시간제로 변환                        
                        date = self.process_time_to_24hr_format(temp_date, time_12hr)
                        # 사용자 ID 비식별화
                        user = self.anonymize_user_id(user_real)
                        
                        content = f'"User: {user}, Message: {message}'
                        
                        metadata = {
                            "date":  date.strftime("%Y-%m-%d %H:%M:%S"),
                            "year": date.year,
                            "month": date.month,
                            "day": date.day,
                            "user": user,
                            "row": i,
                            "source": str(self.file_path),
                        }
                        i += 1 # 행 번호 증가
                        yield Document(page_content=content, metadata=metadata)
       
        
        # NOTE - choh(2024.04.05) - 기존 코드, csv 파일인 경우
        else:
            df = pd.read_csv(csvfile)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Date_strf"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").astype(str)
            for i, row in df.iterrows():
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
