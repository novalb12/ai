import cv2
import numpy as np

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype=np.float32)

k = np.arange(10)
cells_labels = np.repeat(k, 250)


test_digits = np.vsplit(test_digits, 50)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32)


# KNN

def cari(img) :
	global cells
	global cells_labels
	knn = cv2.ml.KNearest_create()
	knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
	ret, result, neighbours, dist = knn.findNearest(img, k=3)
	return result

def show_webcam(mirror=True):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mirror:     
            img = cv2.flip(img, 1)
        gray=cv2.resize(gray, (28,28))
        print(gray.shape)
        print(cari(gray))
        #text = get_string(img)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #if(len(text)> 0):
            #print(text)
            #cv2.putText(img,text,(0,10),font,0.5,(255,255,255),2,cv2.LINE_AA)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        cv2.imshow('my webcam', gray)
    cv2.destroyAllWindows()
    cam.release()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()

