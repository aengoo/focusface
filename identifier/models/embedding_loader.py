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

        :param embed_db_path:
        :param idt_model:
        :param n_faces:
        """
        self.ebd_dict = {}  # destination

        if os.path.isfile(embed_db_path):
            self.df = pd.read_csv(embed_db_path, dtype=db_column_configs)
        else:
            print('[ERROR] Can Not Find Embedding DB CSV File')
            exit()

        self.df = self.df.fillna('N/A')
        self.df.insert(self.df.shape[1], 'CHECKSUM', [False] * self.df.shape[0], True)
        self._model = idt_model
        # added column 'CHECKSUM', all rows are initialized as False(bool)

        self._check_db()
        if n_faces:  # n_faces 입력받았으면
            target_number = self.df[self.df['IS_TARGET']].shape[0]
            if n_faces < target_number or n_faces > self.df.shape[0]:
                n_faces = target_number
                print(f"[NOTICE] n_face changed as number of target faces: {target_number}")

            selected = random_combination(self.df[~(self.df['IS_TARGET'])].T.to_dict().items(), n_faces)
            self.df = self.df[self.df['IS_TARGET']].append([i[1] for i in selected], ignore_index=True)
        else:  # n_faces 입력 안받았으면 그냥 전체 다 돌리는걸로 취급, 아무것도안해도될듯
            pass

        self._regen_embeddings()
        self._read_n_ebd()

    def _check_db(self):
        for idx, row in self.df.iterrows():
            self.df.at[idx, 'CHECKSUM'] = sha256_checksum(row['MUGSHOT_PATH'], self._model)

    def _regen_embeddings(self):
        print("Generating(Checking) Face Embeddings...", end="")
        for cnt in range(3):
            for mugshot_path in self.df[~(self.df['CHECKSUM'])]['MUGSHOT_PATH'].values:
                file_name, file_dir = (os.path.split(mugshot_path)[-1], os.path.join(*os.path.split(mugshot_path)[:-1]))
                file_id = file_name[:file_name.rindex('.')]
                mugshot_img = face_recognition.load_image_file(mugshot_path)
                encodings = face_recognition.face_encodings(mugshot_img, model=self._model)
                if len(encodings) > 1:
                    print(f'[WARNING] multiple face embedding output from image \"{mugshot_path}\".')
                with open(os.path.join(file_dir, '.'.join([file_id, self._model, 'ebd'])), 'w') as ebd:
                    ebd.write(','.join([str(v) for v in encodings[0].tolist()]))
                sha256_encrypt(mugshot_path, self._model)
            self._check_db()  # checksum verification again
            if not self.df[~(self.df['CHECKSUM'])].shape[0]:
                break
            else:
                print(f'[NOTICE] there is any unverified checksum. trying again...({cnt}/3)')
        print("Complete!!")

    def _read_n_ebd(self):
        print("Loading Embeddings...", end="")
        for i in self.df['ID'].values:
            mugshot_path = self.df[self.df['ID'] == i]['MUGSHOT_PATH'].values[0]
            ebd_path = '.'.join(mugshot_path.split('.')[:-1] + [self._model + '.ebd'])
            with open(ebd_path, 'r') as e:
                line = e.read()
                ebd_arr = np.array([float(t) for t in line.replace('\n', '').split(',')], dtype='float64')
                self.ebd_dict.update({i: ebd_arr})
        print("Complete!!")


def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def sha256_checksum(img_path: str, ext: str):
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
    ebd_path = '.'.join(img_path.split('.')[:-1] + [ext + '.ebd'])
    checksum_path = '.'.join(img_path.split('.')[:-1] + [ext + '.checksum'])
    with open(img_path, "rb") as f1, open(ebd_path, "rb") as f2, open(checksum_path, "w") as c:
        b1 = f1.read()  # read entire file as bytes
        b2 = f2.read()  # read entire file as bytes
        checksum_hash = hashlib.sha256(b1 + b2).hexdigest()
        c.write(checksum_hash)


def create_seed(base_path: str, target_faces_dir: str, sample_faces_dir: str, dst_file_name: str = 'seed'):
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
