import sys
from PyQt5.QtWidgets import QApplication
from viewer import LightViewer

def main():
	app = QApplication(sys.argv)
	viewer = LightViewer()
	viewer.show()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main()