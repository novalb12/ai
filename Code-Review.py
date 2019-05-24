'''
	Biasakan Tulis Logic dari Code di File ini apa
	Logic File :
		-> Buat Model dengan menggunakan KNN
		-> Gunakan model untuk prediksi
'''
import numpy as np
import cv2

'''	
	handle data train section
	Kalau butuh analisa tulis, semisal
	img contain 
		50 row and 50 col
		each 5 line(in arrow) contain same number
'''
#read img and change to grayscale
digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)

#get matrix for each line
rows = np.vsplit(digits, 50)

# DATA EXTRACT SECTION
# get and labeled matrix image each number
cells = []
for row in rows:
	# get matrix image number for each col in line
	row_cells = np.hsplit(row, 50)
	for cell in row_cells:
		# change matrix 2d to 1d
		cell = cell.flatten()
		# init data for data train
		cells.append(cell)
# change cells item data type to float32
cells = np.array(cells, dtype=np.float32)

# make label for each data
# make matrix 0 - 9
k = np.arange(10)

# rec for 250 data
cells_labels = np.repeat(k,250)

# make KNN model
# model was saved at variabel "knn"
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)


'''
	PREDICT SECTION
	Logic :
		-> image from webcam resize to (20,20)
			-> (20, 20) is "cell" shape
		-> change to gray image
			-> flat image(2d) to 1d matrix
		-> do predict
		-> print (result)
'''

cap = cv2.VideoCapture(0)


while (cap.isOpened()):
	ret, img = cap.read()
	img = cv2.flip(img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (20,20))
	data_predict = np.copy(gray)
	data_predict = [data_predict.flatten()]
	data_predict = np.array(data_predict, dtype=np.float32)
	_, result, _, _ = knn.findNearest(data_predict, k=3)
	print( result )
	if cv2.waitKey(1) == 27:
		breal
	cv2.imshow('webcam', gray)

cv2.destroyAllWindows()
cap.release()

