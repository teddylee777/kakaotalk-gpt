import pytest
from datetime import datetime
from kakaotalk_loader import KaKaoTalkLoader
import pandas as pd

def test_process_time_to_24hr_format():
    loader = KaKaoTalkLoader(file_path="dummy_path", file_suffix=".txt")
    date_obj = datetime(2024, 4, 5)

    # 오전 시간 테스트
    assert loader.process_time_to_24hr_format(date_obj, "오전 11:23") == datetime(2024, 4, 5, 11, 23)

    # 오후 시간 테스트 (오후 12시 제외)
    assert loader.process_time_to_24hr_format(date_obj, "오후 1:23") == datetime(2024, 4, 5, 13, 23)

    # 오후 12시 테스트
    assert loader.process_time_to_24hr_format(date_obj, "오후 12:23") == datetime(2024, 4, 5, 12, 23)

    # 오전 12시 테스트
    assert loader.process_time_to_24hr_format(date_obj, "오전 12:00") == datetime(2024, 4, 5, 0, 0)

def test_process_date():
    loader = KaKaoTalkLoader(file_path="dummy_path", file_suffix=".txt")

    # 정상적인 날짜 문자열 테스트
    is_parse, date = loader.process_date("-------- 2024년 4월 5일 화요일 --------")
    assert is_parse, date == pd.to_datetime("2024-04-05")

    # 잘못된 날짜 문자열 테스트
    is_parse, date = loader.process_date("This is not a date")
    assert is_parse == False, date == "This is not a date"

def test_txt___read_file(mocker):
    # 가상의 파일 내용
    fake_file_content = [
        "LLM RAG Langchain 통합 님과 카카오톡 대화\n",
        "저장한 날짜 : 2024-04-05 01:36:14\n",
        "\n",
        "--------------- 2024년 3월 27일 수요일 ---------------\n",
        "TEST님이 들어왔습니다.타인, 기관 등의 사칭에 유의해 주세요. 금전 또는개인정보를 요구 받을 경우 신고해 주시기 바랍니다.운영정책을 위반한 메시지로 신고 접수 시 카카오톡 이용에 제한이 있을 수 있습니다. \n",
        "불법촬영물등 식별 및 게재제한 조치 안내\n",
        "그룹 오픈채팅방에서 동영상・압축파일 전송 시 전기통신사업법에 따라 방송통신심의위원회에서 불법촬영물등으로 심의・의결한 정보에 해당하는지를 비교・식별 후 전송을 제한하는 조치가 적용됩니다. 불법촬영물등을 전송할 경우 관련 법령에 따라 처벌받을 수 있사오니 서비스 이용 시 유의하여 주시기 바랍니다.\n",
        "성민상님이 들어왔습니다.\n",
        "[가나다] [오전 10:55] 안녕하세요\n",
        "[가나다] [오전 10:55] RAG관련해서 질문 해도될까요\n",
        "[가나다] [오전 10:57] 혹시 한국어에 유리한 임베딩 방법이 있을가요?\n",
        "[J] [오전 11:00] Bge m3 모델이 잘합니다\n",
        "[가나다] [오전 11:01] 오우 감사합니다 \n",
        "rag 입문인데\n",
        "[가나다] [오전 11:01] 경우의수가 너무 많네요\n",
        "[ABC] [오전 11:05] OPENAI 임베딩 쓰는 것보다 효과가 좋은 것인가요?\n",
        "[DEF] [오전 11:06] 온프레미스로 돌릴 목적이신가…\n",
        "[GHF] [오전 11:06] https://huggingface.co/BAAI/bge-m3\n",
        "[GHF] [오전 11:07] multilingual이라고 써있긴한대 한국어 임베딩 성능도 잘 나오려나요\n",
        "[1234] [오전 11:08] 현존하는 것 중에선 한국어 임베딩이 제일 좋은 것 같아요. 도메인 별로 다를 수도 있으니 직접 본인 데이터셋으로 해보셔서 테스트 해보세요\n",
        "[1234] [오전 11:26] bge m3 임베딩 전에 꼭 전처리 해야 할 팁이 있을까요? 그냥 문장 넣어도 의미적으로 잘 만드는 것일지..\n",
        "[J] [오전 11:26] 그냥 넣음됩니다 html 태그 같은건 빼는게 좋겠군요\n",
        "[가나다] [오전 11:29] 크 모두 감사합니다 \n",
    ]
    
    # 파일 읽기를 위한 모의 설정
    mocker.patch('builtins.open', mocker.mock_open(read_data="\n".join(fake_file_content)))
    
    loader = KaKaoTalkLoader(file_path="dummy_path", file_suffix=".txt")
    documents = list(loader._read_file_test(open("dummy_path")))
    
    # 기대되는 출력 확인
    # 예제에서는 단순히 문서의 개수와 첫 번째 문서의 날짜를 확인합니다.
    # 실제 테스트에서는 더 세밀한 조건을 확인해야 할 수 있습니다.
    expected_number_of_documents = 14  # 가상의 파일 내용에 따라 조정
    assert len(documents) == expected_number_of_documents
    
    # 첫 번째 대화의 날짜와 메시지 내용 확인
    assert documents[0].metadata['date'] == "2024-03-27 10:55:00"
    assert documents[0].page_content == '"User: **, Message: 안녕하세요'

