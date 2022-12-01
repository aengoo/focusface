import hashlib
import pandas as pd
import numpy as np
import random
import face_recognition
import os

db_column_configs = {
    'ID': 'object',
    'NAME': 'object',
    'SEX': 'object',
    'BIRTH': 'object',
    'RAP': 'object',
    'DETAIL': 'object',
    'MUGSHOT_PATH': 'object',
    'IS_TARGET': 'bool'
}

image_ext_list = ['jpg', 'png']


class EmbeddingLoader:
    def __init__(self, embed_db_path: str = '', idt_model: str = 'small', n_faces: int = 0):
        """
        식별을 위한 얼굴 임베딩 벡터를 생성하고 관리합니다.
        :param embed_db_path: 임베딩 DB CSV 파일 경로
        :param idt_model: small or large, 얼굴 인코더의 크기입니다. 랜드마크의 갯수와 CNN 모델 사이즈에 차이가 있습니다. 성능에 큰 영향을 미치지 않습니다.
        :param n_faces: 대조할 얼굴의 개수를 입력합니다. 실 사용에서는 타깃과 같은 수로 두는 것이 일반적입니다.
        """
        self.ebd_dict = {}  # destination

        if os.path.isfile(embed_db_path):
            self.df = pd.read_csv(embed_db_path, dtype=db_column_configs)  # CSV 파일을 pandas 데이터프레임 객체로 읽어들입니다.
        else:
            print('[ERROR] Can Not Find Embedding DB CSV File')
            exit()

        self.df = self.df.fillna('N/A')  # 데이터 프레임의 비어있는 셀을 'N/A'로 채웁니다.

        # 무결성 검사 결과 기록을 위한 열(column)을 생성합니다.
        self.df.insert(self.df.shape[1], 'CHECKSUM', [False] * self.df.shape[0], True)

        # 모델에 따라 임베딩 벡터가 다르기 때문에 별개로 기록합니다.
        self._model = idt_model
        # added column 'CHECKSUM', all rows are initialized as False(bool)

        self._check_db()  # 먼저 무결성 검사를 한번 실시하여 기록합니다.

        if n_faces:  # n_faces 입력받았으면
            target_number = self.df[self.df['IS_TARGET']].shape[0]
            if n_faces < target_number or n_faces > self.df.shape[0]:
                # n_faces가 이상하게 세팅되어 있으면 타깃 수와 일치하도록 수정합니다.
                n_faces = target_number
                print(f"[NOTICE] n_face changed as number of target faces: {target_number}")

            # n_faces에 따라 랜덤 조합을 선정합니다.
            selected = random_combination(self.df[~(self.df['IS_TARGET'])].T.to_dict().items(), n_faces)
            # n_faces에 맞게 선정된 랜덤 조합에 따라 데이터프레임을 재구성합니다.
            self.df = self.df[self.df['IS_TARGET']].append([i[1] for i in selected], ignore_index=True)
        else:  # n_faces 입력 안받았으면 그냥 전체 다 돌리는걸로 취급합니다.
            pass

        self._regen_embeddings()  # 무결성 검사를 통과하지 못했거나, 아직 임베딩이 생성되지 않은 이미지들의 임베딩을 생성합니다.
        self._read_n_ebd()  # 임베딩 벡터를 로드합니다.

    def _check_db(self):
        """
        임베딩이 변경되었는지 무결성 검사를 실시하고, 그 결과를 데이터프레임의 'CHECKSUM' 행에 기록합니다.
        :return: 반환값이 없습니다.
        """
        for idx, row in self.df.iterrows():
            self.df.at[idx, 'CHECKSUM'] = sha256_checksum(row['MUGSHOT_PATH'], self._model)

    def _regen_embeddings(self):
        """
        데이터프레임의 'CHECKSUM' 행을 확인하여 무결성 검사를 통과하지 못한 얼굴 임베딩을 재생성합니다.
        이 과정에서 아직 임베딩이 생성되지 않은 경우도 처리합니다.
        :return: 반환값이 없습니다.
        """
        print("Generating(Checking) Face Embeddings...", end="")
        for cnt in range(3):  # 최대 3번까지 반복합니다. 보통 단번에 끝납니다.
            for mugshot_path in self.df[~(self.df['CHECKSUM'])]['MUGSHOT_PATH'].values:
                # 무결성 검사 미통과한 얼굴 영상 패스들로 반복

                file_name, file_dir = (os.path.split(mugshot_path)[-1], os.path.join(*os.path.split(mugshot_path)[:-1]))
                file_id = file_name[:file_name.rindex('.')]  # 파일 확장자 제거 처리

                # OPEN_CV는 기본 세팅에서 BGR 순으로 이미지를 읽습니다.
                # face_recognition 라이브러리는 이를 역순으로 뒤집어 RGB로 읽습니다.
                # 추후 수정시 염두해야 합니다.
                mugshot_img = face_recognition.load_image_file(mugshot_path)

                # 영상으로부터 얼굴 임베딩을 추출합니다.
                encodings = face_recognition.face_encodings(mugshot_img, model=self._model)

                if len(encodings) > 1:  # 임베딩이 여러개 추출되었으면 경고, 해당 영상 확인 필요
                    print(f'[WARNING] multiple face embedding output from image \"{mugshot_path}\".')

                # 임베딩 파일을 저장합니다.
                with open(os.path.join(file_dir, '.'.join([file_id, self._model, 'ebd'])), 'w') as ebd:
                    ebd.write(','.join([str(v) for v in encodings[0].tolist()]))

                sha256_encrypt(mugshot_path, self._model)  # 얼굴 이미지와 임베딩 SHA-256을 서로 합쳐서 체크섬을 생성합니다.

            self._check_db()  # checksum verification again

            # 재검, 무결성 검사 미통과 데이터가 있는지
            if not self.df[~(self.df['CHECKSUM'])].shape[0]:
                break
            else:
                print(f'[NOTICE] there is any unverified checksum. trying again...({cnt}/3)')
        print("Complete!!")

    def _read_n_ebd(self):
        """
        얼굴 매칭(비교)를 위해 임베딩 데이터를 numpy 배열로 메모리에 로드합니다.
        :return: 반환값이 없습니다.
        """
        print("Loading Embeddings...", end="")
        for i in self.df['ID'].values:
            mugshot_path = self.df[self.df['ID'] == i]['MUGSHOT_PATH'].values[0]
            ebd_path = '.'.join(mugshot_path.split('.')[:-1] + [self._model + '.ebd'])
            with open(ebd_path, 'r') as e:
                line = e.read()
                ebd_arr = np.array([float(t) for t in line.replace('\n', '').split(',')], dtype=np.float64)
                self.ebd_dict.update({i: ebd_arr})
        print("Complete!!")


def random_combination(iterable, r):
    """
    랜덤 조합을 생성합니다.
    :param iterable: iterable 객체
    :param r: 조합을 구성할 개체의 수
    :return: 생성된 랜덤 조합 데이터를 반환합니다.
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def sha256_checksum(img_path: str, ext: str):
    """
    단일 영상과 그 임베딩 파일에 대한 무결성 검사를 실행합니다.
    :param img_path: 이미지 경로
    :param ext: 확장자명, 여기서는 인코더 모델의 크기(small or large)를 의미합니다.
    :return: 무결성 검사 결과를 반환합니다.
    """
    ebd_path = '.'.join(img_path.split('.')[:-1] + [ext + '.ebd'])
    checksum_path = '.'.join(img_path.split('.')[:-1] + [ext + '.checksum'])
    if os.path.isfile(ebd_path) and os.path.isfile(checksum_path) and os.path.isfile(img_path):
        try:
            with open(img_path, "rb") as f1, open(ebd_path, "rb") as f2, open(checksum_path, "r") as c:
                b1 = f1.read()  # read entire file as bytes
                b2 = f2.read()  # read entire file as bytes
                readable_hash = hashlib.sha256(b1 + b2).hexdigest()
                checksum_hash = c.read()
                return readable_hash == checksum_hash
        except IOError:
            print('[I/O ERROR] sha256 checksum')
    else:
        return False


def sha256_encrypt(img_path: str, ext: str):
    """
    단일 영상과 그 임베딩 파일을 각각 SHA-256 암호화하여, 이를 합친 CHECKSUM 파일을 생성합니다.
    :param img_path: 이미지 경로
    :param ext: 확장자명, 여기서는 인코더 모델의 크기(small or large)를 의미합니다.
    :return: 반환값이 없습니다.
    """
    ebd_path = '.'.join(img_path.split('.')[:-1] + [ext + '.ebd'])
    checksum_path = '.'.join(img_path.split('.')[:-1] + [ext + '.checksum'])
    with open(img_path, "rb") as f1, open(ebd_path, "rb") as f2, open(checksum_path, "w") as c:
        b1 = f1.read()  # read entire file as bytes
        b2 = f2.read()  # read entire file as bytes
        checksum_hash = hashlib.sha256(b1 + b2).hexdigest()
        c.write(checksum_hash)


def create_seed(base_path: str, target_faces_dir: str, sample_faces_dir: str, dst_file_name: str = 'seed'):
    """
    본 얼굴 인식 시스템은 식별하고자 하는 대상들의 얼굴 이미지가 저장된 DB가 필요합니다.
    뿐만 아니라, 해당 인물의 정보까지 읽어 들일 수 있어야합니다.
    이 작업을 원활하게 수행하기 위해, 현재 버전의 설계대로 디렉토리 구조를 엄격하게 준수해야 합니다.
    이는 현재 경로의 README.md를 참고하기 바랍니다.
    디렉토리 구조를 준수하여 구성이 완료되었다면, 코드를 실행하여 CSV 파일을 생성해야 합니다.
    프로젝트 루트 디렉토리에 예제 코드가 있습니다.

    :param base_path: 디렉토리 구조상 기본이 되는 디렉토리 경로를 입력합니다. 예제 코드 참고
    :param target_faces_dir: 식별하고자 하는 얼굴들이 있는 디렉토리 경로
    :param sample_faces_dir: 허수 디렉토리 경로
    :param dst_file_name: 생성할 CSV 파일명
    :return: 반환값이 없습니다.
    """
    print("Generating Suspect DB Seed...", end="")
    target_faces_path = os.path.join(base_path, target_faces_dir)
    sample_faces_path = os.path.join(base_path, sample_faces_dir)
    tdf = pd.DataFrame(columns=list(db_column_configs.keys()))
    # RAP : Record of Arrests and Prosecutions

    f_list_dict = {}
    if os.path.isdir(target_faces_path) and os.path.isdir(sample_faces_path):
        f_list_dict.update({'target': [target_faces_path, os.listdir(target_faces_path)]})
        f_list_dict.update({'sample': [sample_faces_path, os.listdir(sample_faces_path)]})
    else:
        # TODO: 타겟 패스랑 샘플 패스가 없는경우
        pass

    id_counter = 0
    for cls in f_list_dict.keys():
        for f in f_list_dict[cls][1]:
            fd_path = os.path.join(f_list_dict[cls][0], f)
            f_name = ''
            for n in os.listdir(fd_path):
                if n.split('.')[-1] in image_ext_list:
                    f_name = n
            f_path = os.path.join(fd_path, f_name)
            new_row = {'ID': str(id_counter).zfill(8),
                       'NAME': f,
                       'MUGSHOT_PATH': f_path,
                       'IS_TARGET': cls == 'target',
                       }
            tdf = tdf.append(new_row, ignore_index=True)
            id_counter += 1
    tdf.to_csv(os.path.join(base_path, dst_file_name), index=False)
    print("Complete!!")
