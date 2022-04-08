import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtCore
from general import *
from cmdl_stream_server import *

form_class = uic.loadUiType("./focusface.ui")[0]


class OptStruct:
    def __init__(self, **entries):
        d = {}
        for k in entries.keys():
            d.update({k.replace('-', '_'): entries[k]})
        self.__dict__.update(d)


class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.opt_dict = {}
        self.ui_dict = {
            'data': self.lineEdit_1_1,
            'vid-res': self.comboBox_1_2,
            'det-model': self.comboBox2_1,
            'det-weight': self.lineEdit_2_2,
            'box-ratio': self.doubleSpinBox_2_3,
            'down': self.spinBox_2_4,
            'conf-thresh': self.doubleSpinBox_2_5,
            'suspect-db': self.lineEdit_3_1,
            's-faces': self.lineEdit_3_2,
            'n-faces': self.spinBox_3_3,
            'idt-model': self.comboBox_3_4,
            'iou-thresh': self.doubleSpinBox_4_1,
            'insense': self.spinBox_4_2,
            'criteria': self.doubleSpinBox_4_3,
            'redis-port': self.spinBox_5_1,
        }

        self.reset_arguments()
        self.modeRadioButton_1.clicked.connect(self.mode_selected_adapt)
        self.modeRadioButton_2.clicked.connect(self.mode_selected_adapt)
        self.outputRadioButton_1.clicked.connect(self.output_selected_adapt)
        self.outputRadioButton_2.clicked.connect(self.output_selected_adapt)
        self.actionReset_Arguments.triggered.connect(self.reset_arguments)
        self.lockOnButton.clicked.connect(self.lock_on_settings)
        self.comboBox_1_2.activated.connect(self.res_selected_adapt)
        self.comboBox_1_2.activated.connect(self.down_selected_adapt)
        self.spinBox_2_4.valueChanged.connect(self.down_selected_adapt)
        self.progressBar.setValue(0)
        self.runPushButton.clicked.connect(self.run_selected_process)

    def reset_arguments(self):
        """
        TODO// yaml 이나 config python 파일에서 기본세팅 읽어오기
        argparse에서도 호환가능한 파일로 구성하거나, 여기있는 옵션을 메인 동작 코드에 넘겨줄수 있어야 함.
        """
        default_opt_dict = cfg_opt_dict
        for k in self.ui_dict.keys():
            if isinstance(self.ui_dict[k], QLineEdit):
                self.ui_dict[k].setText(default_opt_dict[k])
            elif isinstance(self.ui_dict[k], QComboBox):
                self.ui_dict[k].setCurrentText(default_opt_dict[k])
            elif isinstance(self.ui_dict[k], QSpinBox):
                self.ui_dict[k].setValue(default_opt_dict[k])
            elif isinstance(self.ui_dict[k], QDoubleSpinBox):
                self.ui_dict[k].setValue(default_opt_dict[k])
        self.mode_selected_adapt()
        self.output_selected_adapt()
        self.res_selected_adapt()
        self.down_selected_adapt()

    def read_all_ui(self):
        for k in self.ui_dict.keys():
            if isinstance(self.ui_dict[k], QLineEdit):
                self.opt_dict.update({k: self.ui_dict[k].text()})
            elif isinstance(self.ui_dict[k], QComboBox):
                self.opt_dict.update({k: combo_matcher[k][self.ui_dict[k].currentText()]})
            elif isinstance(self.ui_dict[k], QSpinBox):
                self.opt_dict.update({k: self.ui_dict[k].value()})
            elif isinstance(self.ui_dict[k], QDoubleSpinBox):
                self.opt_dict.update({k: self.ui_dict[k].value()})

    def run_selected_process(self):
        if self.lockOnButton.isChecked():
            if self.modeRadioButton_1.isChecked():
                self.run_stream_process()
            else:
                pass  # TODO// evaluator 구현 필요

    def run_stream_process(self):
        """
        TODO// 스트림 프로세스 실행 (라디오 버튼 옵션 체크 하고 ㄱㄱ해야함)
        """
        run_stream_server(OptStruct(**self.opt_dict))

    def res_selected_adapt(self):
        if str(self.comboBox_1_2.currentText()) == 'adaptive (initialized by data)':
            self.spinBox_2_4.setValue(1)
            self.spinBox_2_4.setDisabled(True)
        else:
            self.spinBox_2_4.setEnabled(True)

    def down_selected_adapt(self):
        vid_res_text = str(self.comboBox_1_2.currentText())
        if vid_res_text != 'adaptive (initialized by data)':
            res = AVAILABLE_RESOLUTIONS[qt2opt_vid_res[vid_res_text]]
            self.label_2_4.setText(str(res[1]//self.spinBox_2_4.value()) + '×' + str(res[0]//self.spinBox_2_4.value()))
        else:
            self.label_2_4.setText('-')

    def mode_selected_adapt(self):
        """
        TODO// evaluation mode 구현시 작동 - UI 추가 및 모드 선택에 따른 일부 옵션 비활성화
        """
        pass

    def output_selected_adapt(self):
        if self.outputRadioButton_1.isChecked():  # opencv 출력 설정
            self.spinBox_5_1.setDisabled(True)
            self.opt_dict.update({'output': 'opencv'})
        elif self.outputRadioButton_2.isChecked():  # Redis 출력 설정
            self.spinBox_5_1.setEnabled(True)
            self.opt_dict.update({'output': 'redis'})

    def lock_on_settings(self):
        if self.lockOnButton.isChecked():
            self.optFrame.setDisabled(True)
            self.modeGroupBox.setDisabled(True)
            self.outputGroupBox.setDisabled(True)
            self.runPushButton.setEnabled(True)
            self.read_all_ui()
        else:
            self.optFrame.setEnabled(True)
            self.modeGroupBox.setEnabled(True)
            self.outputGroupBox.setEnabled(True)
            self.runPushButton.setDisabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()
