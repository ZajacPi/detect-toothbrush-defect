import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy

img = cv2.imread("lena.png")

if img is None:
    print("Błąd: nie wczytano obrazu!")
else:
    print("Obraz wczytany poprawnie")
    cv2.imshow("Lena", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()