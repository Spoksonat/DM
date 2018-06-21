#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

//-------------------------Constantes-------------------------//
#define Nx 1024
#define Nv 1024

#define L 2.0 //1.0
#define L_min -1.0 // -0.5
#define V 2.0
#define V_min -1.0

#define pi 3.141592654
#define FLOAT float

#define T 321
#define skip 20
#define deltat 0.1

//-------------------------Variables globales-------------------------//
FLOAT delx=L/(Nx);
FLOAT delv=V/(Nv);
FLOAT L_max = L_min+L;
FLOAT V_max = V_min+V;

int i,j,k;
int i_v_new, j_x_new;
FLOAT x, v, x_new, v_new;

FILE *phase_rela_dat, *phase_four_dat, *dens_dat, *acc_dat, *pot_dat, *vels_dat;
FLOAT *phase, *phase_new, *dens, *acc, *pot, *pot_temp, *vels;

FLOAT Kx;
FLOAT kx;
fftw_complex *rho_out, *rho_in, *rho_fin;
fftw_plan rho_plan;

char *method;

//-------------------------Declaracion de funciones-------------------------//
void gauss(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT amp, FLOAT sigma);
void bullet(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT amp1, FLOAT sigma1, FLOAT x1, FLOAT amp2, FLOAT sigma2, FLOAT x2);
void jeans(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT rho, FLOAT amp, FLOAT sig, int n);
void densidad(FLOAT *fase, FLOAT *rho);
void potential(FLOAT *rho, FLOAT *Va, FLOAT *V_temp);
void potfourier_real(FLOAT *rho, FLOAT *res);
void acceleration(FLOAT *Va, FLOAT *aceleracion);
void update(FLOAT * fase, FLOAT * azz, FLOAT * phase_temp);
int ndx(int fila, int column);
void printINFO(int indice, FLOAT * density, FILE * dens_file, FLOAT * azz, FILE * azz_file, FLOAT * potencial, FILE * pot_file, FLOAT * fase, FILE * fase_file, FLOAT * speed, FILE * speed_file);
void printCONS(char *state);
FLOAT sinc(FLOAT x);
void check(FLOAT *arreglo);
void check2(fftw_complex *arreglo);
void dens_vel(FLOAT *fase, FLOAT *rho_v);
void RELAX();
void FOURIER();
//-------------------------Main-------------------------//
int main(){
  dens_dat=fopen("dens_dat.txt", "w");
  acc_dat=fopen("acc_dat.txt", "w");
  pot_dat=fopen("pot_dat.txt", "w");
  vels_dat=fopen("vels_dat.txt", "w");

  phase = malloc(sizeof(FLOAT)*Nx*Nv);
  phase_new = malloc(sizeof(FLOAT)*Nx*Nv);
  dens=malloc(sizeof(FLOAT)*Nx);
  acc=malloc(sizeof(FLOAT)*Nx);
  pot=malloc(sizeof(FLOAT)*Nx);
  pot_temp=malloc(sizeof(FLOAT)*Nx);
  vels=malloc(sizeof(FLOAT)*Nv);
  check(phase); check(phase_new); check(dens); check(acc); check(pot); check(pot_temp); check(vels);

  //gauss(phase, phase_new, 4, 0.08);
  bullet(phase, phase_new, 5, 0.04, -0.45, 5, 0.04, 0.45);
  //jeans(phase, phase_new, 4.0, 4.0, 0.25, 2);

  //RELAX();
  FOURIER();

  return 0;
}

//-------------------------Funciones-------------------------//

// Función f(x,v) de densidad de espacio de fase gaussiana
void gauss(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT amp, FLOAT sigma){
// Recorre todo el espacio discreto de velocidades y posiciones
  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      // Avanza en pasos de (j*deltax) sobre el eje de posiciones desde el punto (L_min + 0.5*deltax),no desde el borde (L_min).
      FLOAT pos=L_min+j*delx+0.5*delx;
      // Avanza en pasos de (i*deltav) sobre el eje de velocidades desde el punto (V_min + 0.5*deltav),no desde el borde (V_min).
      FLOAT vel=V_min+i*delv+0.5*delv;
      // Asigna los valores de la gaussiana al puntero arreglo[]. Llamando a los elementos de este puntero a_{v,x} con {v,x} las ubicaciones en la cuadrícula de velocidades y posiciones, este puntero tiene asignados datos de la forma arreglo = [a_{0,0},a_{0,1},...,a_{0,Nx}, a_{1,0}, a_{1,1}, ..., a_{1,Nx},........a_{Nv,0},a_{Nv,1},...a_{Nv,Nx}]
      arreglo[ndx(i,j)]=amp*exp(-(pow(pos,2)+pow(vel,2))/sigma);
    }
  }
}

// Funcion f(x_{g1},v_{g1},x_{g2},v_{g2}) de densidad de espacio de fase para el bullet cluster ({g1} = galaxia 1, {g2}= galaxia 2), con parámetros de entrada las posiciones iniciales de cada una de las dos galaxias(x_1,x_2), las desviaciones estandar de cada gaussiana que representan a las galaxias (sigma1 para {g1} y sigma2 para {g2}) y las amplitudes de cada gaussiana (amp1, amp2). Se asumen las velocidades iniciales promedio iguales a cero.
void bullet(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT amp1, FLOAT sigma1, FLOAT x1, FLOAT amp2, FLOAT sigma2, FLOAT x2){
  // Recorre todo el espacio discreto de velocidades y posiciones
  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      // Avanza en pasos de (j*deltax) sobre el eje de posiciones desde el punto (L_min + 0.5*deltax),no desde el borde (L_min).
      FLOAT pos=L_min+j*delx;
       // Avanza en pasos de (i*deltav) sobre el eje de velocidades desde el punto (V_min + 0.5*deltav),no desde el borde (V_min).
      FLOAT vel=V_min+i*delv;
      // Asigna los valores de la gaussiana al puntero arreglo[]. Llamando a los elementos de este puntero a_{v,x} con {v,x} las ubicaciones en la cuadrícula de velocidades y posiciones, este puntero tiene asignados datos de la forma arreglo = [a_{0,0},a_{0,1},...,a_{0,Nx}, a_{1,0}, a_{1,1}, ..., a_{1,Nx},........a_{Nv,0},a_{Nv,1},...a_{Nv,Nx}]
      arreglo[ndx(i,j)]=amp1*exp(-(pow(pos-x1,2)+pow(vel,2))/sigma1)+amp2*exp(-(pow(pos-x2,2)+pow(vel,2))/sigma2);
    }
  }
}

// Funcion f(x,v) de densidad de espacio de fase para Inestabilidad de Jeans, con parámetros de entrada: \rhobar (densidad de fase promedio, o sea, f(x,v) promedio), amp(la amplitud de la gaussiana para las velocidades), sig(la desviacion estandar de las velocidades) y n(numero entero, usado para expresar el numero de onda k como un múltiplo del numero de onda inicial k0 = 2*pi/L (en una caja)).
void jeans(FLOAT *arreglo, FLOAT *arreglo_new, FLOAT rho, FLOAT amp, FLOAT sig, int n){
  // Numero de onda inicial
  FLOAT k0 = 2.0*pi/L;
  // Numero de onda n-esimo
  FLOAT k = n*k0;
  // De aca en adelante el proceso de asignacion de valores para arreglo[] es el mismo que para la gaussiana y el bullet cluster
  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      FLOAT pos=L_min+j*delx;
      FLOAT vel=V_min+i*delv;
      arreglo[ndx(i,j)]=rho/(pow(2*pi*sig*sig,0.5))*exp(-pow(vel,2)/(2*sig*sig))*(1+amp*cos(k*pos));
    }
  }
}

// Se calcula la densidad de masa en cada punto del eje de las posiciones, como la suma(integral discreta) de la densidad en el espacio de fase sobre las velocidades (diferencial -> delv)
void densidad(FLOAT *fase, FLOAT *rho){
  for(i=0;i<Nx;i++){
    rho[i]=0.0;
    for(j=0;j<Nv;j++){
      rho[i]+=fase[ndx(j,i)]*delv;
    }
  }
}

// Aplicacion de la ecuacion (2.2.10) para hallar el potencial por metodo de relajacion (discretizacion de las derivadas).
void potential(FLOAT *rho, FLOAT *Va, FLOAT *V_temp){
  for(i=0;i<Nx;i++){
    V_temp[i]=0.0;
  }
  for(j=0;j<Nx*Nx;j++){
    for(i=1;i<Nx-1;i++){
      Va[i]=0.5*(V_temp[i-1]+V_temp[i+1] - rho[i]*delx*delx);
    }
    // Se imponen condiciones de frontera periodicas
    Va[0]=0.5*(V_temp[Nx-1]+V_temp[1] - rho[i]*delx*delx);
    Va[Nx-1]=0.5*(V_temp[Nx-2]+V_temp[0] - rho[i]*delx*delx);
    for(i=0;i<Nx;i++){
      V_temp[i]=Va[i];
    }
  }
}

// Implementacion de ecuacion (2.2.14)- Metodo para encontrar potencial por transformada de Fourier
void potfourier_real(FLOAT *rho, FLOAT *res){

  rho_in=fftw_malloc(sizeof(fftw_complex)*Nx);
  rho_out=fftw_malloc(sizeof(fftw_complex)*Nx);
  rho_fin=fftw_malloc(sizeof(fftw_complex)*Nx);
  check2(rho_in); check2(rho_fin); check2(rho_out);

  for(i=0;i<Nx;i++){
    rho_in[i]=rho[i];
  }
  rho_plan = fftw_plan_dft_1d(Nx, rho_in, rho_out, 1, FFTW_ESTIMATE);
  fftw_execute(rho_plan);
  fftw_destroy_plan(rho_plan);

  rho_out[0]=0.0;
  for(i=1;i<Nx;i++){
    // Variable de fourier zita = i/L
    kx=2*pi/L*(FLOAT)i;
    Kx=kx*sinc(0.5*kx*delx);
    // Aplicacion de ecuacion (2.2.15)
    rho_out[i]=rho_out[i]/(-pow(Kx,2));
  }
  rho_plan = fftw_plan_dft_1d(Nx, rho_out, rho_fin, -1, FFTW_ESTIMATE);
  fftw_execute(rho_plan);
  fftw_destroy_plan(rho_plan);
  for(i=0;i<Nx;i++){
    res[i]=rho_fin[i]/(Nx);
  }
}

// Calcula la aceleracion ya conocido el potencial
void acceleration(FLOAT *Va, FLOAT *aceleracion){
  /*for(i=0;i<Nx-1;i++){
    aceleracion[i]=-(Va[i+1]-Va[i])/delx;
  }
  aceleracion[Nx-1]=-(Va[0]-Va[Nx-1])/delx;*/
  for(i=1;i<Nx-1;i++){
    // Aceleracion = -grad(Potencial)
    aceleracion[i]=-(Va[i+1]-Va[i-1])/(2*delx);
  }
  // Imposicion de condiciones de frontera periodicas
  aceleracion[0]=-(Va[1]-Va[Nx-1])/(2*delx);
  aceleracion[Nx-1]=-(Va[0]-Va[Nx-2])/(2*delx);
}

// Todos los valores de un puntero se ponen en cero, para una posterior asignacion de nuevos valores para el mismo, actualizando los datos para nue
void update(FLOAT * fase, FLOAT * azz, FLOAT * fase_new){
  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      fase_new[ndx(i,j)]=0.0;
    }
  }

  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      // Valores iniciales de velocidad, sin considerar aceleracion
      v=V_min+i*delv+0.5*delv;
      // Valores iniciales de posicion, sin consierar aceleracion
      x=L_min+j*delx+0.5*delx;
     // V= v_0 + a*t
      v_new=v+deltat*azz[j];
     // X = x_0 + v*t
      x_new=x+deltat*v_new;
      // posicion de casilla(i_v_new,j_x_new) con velocidad v_new y posicion x_new 
      i_v_new= (int) ((v_new-V_min)/delv);
      j_x_new= (int) ((x_new-L_min)/delx);

      if(i_v_new >= 0 && i_v_new < Nv){
        if(j_x_new < 0){
          // Se aplica condicion de periodicidad para casillas en eje de posiciones (x_-j = x_{Nx + j - 1})
          j_x_new = (int) (Nx+j_x_new-1);
        }
        // Se aplica otra condicion de frontera (x_{Nx + j} = x_{j%Nx})
        else if(j_x_new >= Nx){
          j_x_new = (int) (j_x_new%Nx);
        }
        fase_new[ndx(i_v_new, j_x_new)]=fase_new[ndx(i_v_new, j_x_new)]+fase[ndx(i,j)];
      }
    }
  }
  for(i=0;i<Nv;i++){
    for(j=0;j<Nx;j++){
      fase[ndx(i,j)]=fase_new[ndx(i,j)];
    }
  }
}

// Ayuda a organizar los elementos de un puntero como arreglo = [a_{0,0},a_{0,1},...,a_{0,Nx}, a_{1,0}, a_{1,1}, ..., a_{1,Nx},........a_{Nv,0},a_{Nv,1},...a_{Nv,Nx}]
int ndx(int fila, int column){
  return fila*Nx+column;
}

// Se imprime informacion acerca de densidad, aceleracion, potencial, velocidad y funcion de densidad en el espacio de fase
void printINFO(int indice, FLOAT * density, FILE * dens_file, FLOAT * azz, FILE * azz_file, FLOAT * potencial, FILE * pot_file, FLOAT * fase, FILE * fase_file, FLOAT * speed, FILE * speed_file){
  if (indice%skip==0){
    for(i=0;i<Nv;i++){
      for(j=0;j<Nx;j++){
        fprintf(fase_file, "%lf ", fase[ndx(i,j)]);
      }
      fprintf(fase_file, "\n");
    }
    for(j=0;j<Nx;j++){
      fprintf(dens_file, "%lf \n", density[j]);
      fprintf(azz_file, "%lf \n", azz[j]);
      fprintf(pot_file, "%lf \n", potencial[j]);
    }
    for(j=0;j<Nv;j++){
      fprintf(speed_file, "%lf \n", speed[j]);
    }
  }
}

// Se guarda archivo con constantes usadas para ejecucion del codigo
void printCONS(char *state){
  FILE *CONS;
  CONS=fopen("Constantes.txt", "w");
  fprintf(CONS, " Nx= %d\n Nv= %d\n L= %lf\n L_min= %lf\n V= %lf\n V_min= %lf\n T= %d\n skip= %d\n deltat= %lf \n Metodo= %s", Nx, Nv, L, L_min, V, V_min, T, skip, deltat, state);
}

// Funcion sinc(x)
FLOAT sinc(FLOAT x){
  if (x==0){
    return 1.0;
  }
  return sin(x)/x;
}

// Revisa condiciones de archivo de entrada. Excepciones
void check(FLOAT *arreglo){
  if(!arreglo){
    printf("Un arreglo no se definio correctamente \n");
    exit(0);
  }
}

// Revisa condiciones de archivo de entrada para el caso de transformadas de Fourier. Excepciones
void check2(fftw_complex *arreglo){
  if(!arreglo){
    printf("Un arreglo tipo fftw_complex no se definio correctamente \n");
    exit(0);
  }
}

// Se calcula la densidad de masa en cada punto del eje de las velocidades, como la suma(integral discreta) de la densidad en el espacio de fase sobre las posiciones (diferencial -> delx)
void dens_vel(FLOAT *fase, FLOAT *rho_v){
  for(j=0;j<Nv;j++){
    rho_v[j]=0.0;
    for(i=0;i<Nx;i++){
      rho_v[j]+=fase[ndx(j,i)]*delx;
    }
  }
}

// Se ejecuta calculo de densidad de masa, potencial, aceleracion, etc... por metodo de relajacion
void RELAX(){
  phase_rela_dat=fopen("phase_rela_dat.txt", "w");
  method="Relaxation";
  printCONS(method);
  for(k=0;k<T;k++){
    printf("Paso %d/%d \n", k+1, T);
    densidad(phase, dens);
    potential(dens, pot, pot_temp);
    acceleration(pot, acc);

    dens_vel(phase, vels);

    printINFO(k, dens, dens_dat, acc, acc_dat, pot, pot_dat, phase, phase_rela_dat, vels, vels_dat);

    update(phase, acc, phase_new);
  }
}

// Se ejecuta calculo de densidad de masa, potencial, aceleracion, etc... por metodo de transformada de Fourier.
void FOURIER(){
  phase_four_dat=fopen("phase_four_dat.txt", "w");
  method="Fourier";
  printCONS(method);
  for(k=0;k<T;k++){
    printf("Paso %d/%d \n", k+1, T);
    densidad(phase, dens);
    potfourier_real(dens, pot);
    acceleration(pot, acc);

    dens_vel(phase, vels);

    printINFO(k, dens, dens_dat, acc, acc_dat, pot, pot_dat, phase, phase_four_dat, vels, vels_dat);

    update(phase, acc, phase_new);
  }
}
