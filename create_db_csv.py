from identifier.models.embedding_loader import create_seed

"""
본 얼굴 인식 시스템은 식별하고자 하는 대상들의 얼굴 이미지가 저장된 DB가 필요합니다.
뿐만 아니라, 해당 인물의 정보까지 읽어 들일 수 있어야합니다.
이 작업을 원활하게 수행하기 위해, 현재 버전의 설계대로 디렉토리 구조를 엄격하게 준수해야 합니다.
이는 현재 경로의 README.md를 참고하기 바랍니다. 
디렉토리 구조를 준수하여 구성이 완료되었다면, 이 코드를 실행하여 CSV 파일을 생성해야 합니다. 
"""

base = '../data/target'  # 데이터셋 경로
target_face_path = 'faces-17'  # 찾고자 하는 얼굴 경로
sample_face_path = 'faces-400'  # 허수 얼굴 경로
create_seed(base, target_face_path, sample_face_path)  # import 되어 있는 코드에서 설명 참고 바랍니다.
