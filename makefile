all: ejecutar cargar graficar

ejecutar:
	g++ RungeKuttaY.cpp

cargar:
	./a.out

graficar: Runge-Kutta.txt
	python graficaY.py
	
