from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QLabel

class ResultsTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Historical Runs:"))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "Task", "Model", "Accuracy"])
        layout.addWidget(self.table)

    def add_run(self, run_id, task, model, accuracy):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(run_id)))
        self.table.setItem(row, 1, QTableWidgetItem(task))
        self.table.setItem(row, 2, QTableWidgetItem(model))
        self.table.setItem(row, 3, QTableWidgetItem(f"{accuracy:.4f}"))
