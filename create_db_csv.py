from identifier.models.embedding_loader import create_seed

base = '../data/target'  # 데이터셋 경로
target_face_path = 'faces-17'  # 찾고자 하는 얼굴 경로
sample_face_path = 'faces-400'  # 허수 얼굴 경로
create_seed(base, target_face_path, sample_face_path)
