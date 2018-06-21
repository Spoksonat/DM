#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
using namespace std;

float dx = 0.001;
int x = 1001;
int n_puntos = 1000000; //n_puntos = (t/dt)
float pi = 3.1415;
float g_eff = 100.0;
float g = 2.0; 
float Mp = 2.435E18 ;
float m_x = 155.0;
float vacs = 1E-17;


float Yeq(float x)
{
    float res = (g/pow(2*pi,1.5))*(pow(x,1.5))*exp(-x);
    return res;
}

float f_prima(float x, float Y)
{ 
    float res = (m_x*Mp/pi)*pow(x,-2)*pow(90.0/g_eff,0.5)*vacs*(pow(Yeq(x),2)- pow(Y,2)) ;
    return res;
}


void runge_kutta(float *x,float *Y)
{
    Y[0] = Yeq(1);
    x[0] = 1.0;

    for (int i=1; i<n_puntos; i++)
    {
        float k1_Y = dx * f_prima(x[i-1],Y[i-1]);
        float k2_Y = dx * f_prima(x[i-1]+ 0.5*dx, Y[i-1] + 0.5 * k1_Y);
        float k3_Y = dx * f_prima(x[i-1]+ 0.5*dx, Y[i-1] + 0.5 * k2_Y);
        float k4_Y = dx * f_prima(x[i-1]+ dx, Y[i-1] + k3_Y);


        float promedio_Y = (1.0/6.0)*(k1_Y + 2.0*k2_Y + 2.0*k3_Y + k4_Y);
       
        x[i] = x[i-1] + dx;
        Y[i] = Y[i-1] + promedio_Y;

    }
}




void imprimir(float *x, float *Y, float *Yeq)
{
    FILE *pf;
    //Abre archivo
    pf = fopen("Runge-Kutta.txt", "w"); 
    for(int i=0; i<n_puntos; i++) 
    {
        fprintf(pf, "%f %f %f  \n", x[i]/x[0], Y[i]/Y[0], Yeq[i]/Yeq[0]);
    }
    fclose(pf);
}



int main()
{
    float *x = (float*)malloc(n_puntos*sizeof(float)); 
    for (int i=0; i<n_puntos; i++)
        x[i] = 0;

    float *Y = (float*)malloc(n_puntos*sizeof(float)); 
    for (int i=0; i<n_puntos; i++)
        Y[i] = 0;

    float *Yeq2 = (float*)malloc(n_puntos*sizeof(float)); 
    for (int i=0; i<n_puntos; i++)
        Yeq2[i] = 0;


    runge_kutta(x,Y);
   
     for (int i=0; i<n_puntos; i++)
        Yeq2[i] = Yeq(x[i]); 
   
    imprimir(x,Y,Yeq2);

    return 0;
}

