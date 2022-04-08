from typing import Tuple


class SuspectInterface:
    def __init__(self, trk_id: int, face_id: str, init_score: float, bbox: list):
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


class SuspectEntry:
    def __init__(self, thresh: float):
        self.suspect_dict = {}
        self.thresh = thresh

    def register(self, suspect: SuspectFace) -> Tuple[bool, SuspectFace]:
        k = int(suspect)
        if k in self.suspect_dict:
            self.suspect_dict[k] += suspect
            del suspect
        else:
            self.suspect_dict.update({int(suspect): suspect})
        if self.suspect_dict[k] >= self.thresh:
            self.suspect_dict[k].fix()
        return self.suspect_dict[k].is_fixed() and not self.suspect_dict[k].is_reported(), self.suspect_dict[k]


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

