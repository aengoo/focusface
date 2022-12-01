from typing import Tuple


class SuspectInterface:
    """
    용의자 객체 형태를 정의합니다.
    주요 기능을 제외한 연산자 오버라이딩을 위주로 구현한 클래스입니다.
    """
    def __init__(self, trk_id: int, face_id: str, init_score: float, bbox: list):
        """
        :param trk_id: 트래킹 ID (tracking index)
        :param face_id: 얼굴 ID (인물 ID)
        :param init_score: 초기화용 부여 점수
        :param bbox: 탐지 상자 (좌표)
        """
        self._tid = trk_id
        self._fid = face_id
        self._score = init_score
        self._last_box = bbox
        self._signed = False

    def get_score(self) -> float:
        return self._score

    def __str__(self):
        return str(self._tid)

    def __int__(self):
        return self._tid

    # 점수 계산을 위한 연산자 오버라이딩
    def __gt__(self, other: float):
        return self._score > other

    def __lt__(self, other: float):
        return self._score < other

    def __ge__(self, other: float):
        return self._score >= other

    def __le__(self, other: float):
        return self._score <= other

    def is_fixed(self) -> bool:
        return self._signed

    def fix(self):
        if self._fid != '-':
            self._signed = True

    def get_last_box(self) -> Tuple[list]:
        return tuple(self._last_box)

    def get_face_id(self) -> str:
        return self._fid


class SuspectFace(SuspectInterface):
    def __init__(self, trk_id: int, face_id: str, init_score: float, bbox: list):
        super().__init__(trk_id, face_id, init_score, bbox)
        self._reported = False

    def __add__(self, other: SuspectInterface):
        if (not self._signed) and other.get_face_id() != '-':
            if self._fid == other.get_face_id():
                self._score += other.get_score()
            else:
                out = self._score - other.get_score()
                self._fid = other.get_face_id()
                self._score = abs(out)
        return self

    def __eq__(self, other):
        if isinstance(other, int):
            return self._tid == other
        # Comparing with another Test object
        elif isinstance(other, SuspectInterface):
            return self._tid == int(other)

    def set_reported(self):
        if self._signed:
            self._reported = True

    def is_reported(self) -> bool:
        return self._reported

    def get_fid(self):
        return self._fid


class SuspectEntry:
    def __init__(self, thresh: float):
        """
        용의자 목록 관리를 위한 객체입니다.
        :param thresh: 결론 도출을 위한 임계값을 설정합니다. 낮을수록 시스템이 민감하게 작동합니다. 일반적으로 10~20 정도로 세팅하는 것이 적당합니다.
        """
        self.suspect_dict = {}
        self.thresh = thresh

    def register(self, suspect: SuspectFace) -> Tuple[bool, SuspectFace]:
        """
        용의자 객체를 등록합니다.
        :param suspect: 등록할 용의자 객체를 입력받습니다.
        :return: 보고 가능한 상태 여부, 용의자 목록을 반환합니다.
        """
        k = int(suspect)
        if k in self.suspect_dict:
            self.suspect_dict[k] += suspect
            del suspect
        else:
            self.suspect_dict.update({int(suspect): suspect})
        if self.suspect_dict[k] >= self.thresh:
            self.suspect_dict[k].fix()
        return self.suspect_dict[k].is_fixed() and not self.suspect_dict[k].is_reported(), self.suspect_dict[k]

    def get_reported_suspects(self):
        sus_list = []
        for sus in self.suspect_dict.values():
            if sus.is_reported:
                sus_list.append(sus.get_fid())
        return tuple(set(sus_list))


if __name__ == '__main__':
    """
    for unit test
    """
    test_inst_entry = SuspectEntry(5.)

    print([test_inst_entry.register(SuspectFace(i, str(i).zfill(8), i + (i / 10), [1, 1, 1, 1]))[0] for i in range(10)])

    a = SuspectFace(1, "00000001", 0.153, [1, 1, 1, 1])
    b = SuspectFace(2, "00000002", 0.123, [1, 1, 1, 1])
    print()
    print(a > 0.15299)  # True
    print(a > 0.15300)  # False
    print(a > 0.15301)  # False
    print()
    print(a < 0.15299)  # False
    print(a < 0.15300)  # False
    print(a < 0.15301)  # True
    print()
    print(a >= 0.15299)  # True
    print(a >= 0.15300)  # True
    print(a >= 0.15301)  # False
    print()
    print(a <= 0.15299)  # False
    print(a <= 0.15300)  # True
    print(a <= 0.15301)  # True

