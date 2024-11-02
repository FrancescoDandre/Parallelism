#include<iostream>
#include <fstream>
//#include <cstdlib>
#include"CNN.h"
#include<omp.h>
#include <iomanip>
#include <chrono>
using namespace std;
#define ROWS 128
#define COLUMNS 128
#define THREADS 8

#define DIM 512 //per generare matrice e kernel da testare

int main(int argc, char * argv[])  {
    int rows, columns;
    rows = ROWS;
    columns = COLUMNS; 
    matrix  A,B,K;
    ifstream myin;
    ofstream myout;
    if (argc!=3) {
        std::cout << "Usage: ./a.exe matrix.txt kernel.txt";
        exit(0);
    }

    //leggo da file la matrice e il kernel
    //matrice
    myin.open(argv[1],ios::in);
    A = read_matrix_file(myin, rows, columns);
    myin.close();

    B = new float *[rows]; // salvo una copia di A
    for (int i=0;i<rows;i++) {
        B[i] = new float[columns];
    }
    for (int i=0;i<rows;i++) {  //avrei potuto parallelizzare anche questo, ma verificando i tempi per una matrice 128x128 non sembra valerne la pena
        for (int j=0;j<columns;j++) {
            B[i][j]=A[i][j];
        }
    }
    
    //kernel
    myin.open(argv[2],ios::in);
    K = read_matrix_file(myin, 4, 4);
    myin.close();

    omp_set_num_threads(THREADS);// definisco il numero di thread


    // eseguo le 4 operazioni e stampo ogni matrice risultante su un file diverso
    myout.open("MaxPool.txt",ios::out);
    print_matrix_file(myout, MaxPoolP(A,rows,columns), rows/4, columns/4); //la matrice ha dimensione 1/4 di quella di input
    myout.close();

    myout.open("AvgPool.txt",ios::out);
    print_matrix_file(myout, AvgPoolP(A,rows,columns), rows/4, columns/4); //la matrice ha dimensione 1/4 di quella di input
    myout.close();

    ReLU6P(A,rows,columns); //modifica la matrice in input
    myout.open("ReLU6.txt",ios::out);
    print_matrix_file(myout, A, rows, columns); 
    myout.close();

    myout.open("COnvolution.txt",ios::out);
    print_matrix_file(myout, (ConvolutionZeroPaddingP(B, rows, columns,K)), rows, columns); //in input la matrice iniziale, se voglio fare la convoluzione sulla matrice processata dalla ReLU6 inserisco A
    myout.close();

    //codice utilizzato per studiare i tempi di esecuzione
/*
    fstream myapp;
    double startwtime, endwtime;

    generate(DIM);//crea i file "matrice_di_test.txt" e "kernel_di_test.txt" sui quali scrive una matrice randomica di dimensione data in input e un kernel randomico(4x4)
    rows=DIM;
    columns=DIM;

    myin.open("matrice_di_test.txt",ios::in);
    B = read_matrix_file(myin, rows, columns);
    myin.close();

    myin.open("kernel_di_test.txt",ios::in);
    K = read_matrix_file(myin, 4, 4);
    myin.close();


    //tempo esecuzione max pool
    startwtime = omp_get_wtime();
    MaxPoolP(B,rows,columns);
    endwtime = omp_get_wtime();
    myapp.open("tempi_esecuzione.txt",ios::out|ios::app);
    myapp<<endl<<endwtime-startwtime<<" ";
    myapp.close();

    startwtime = omp_get_wtime();
    AvgPoolP(B,rows,columns);
    endwtime = omp_get_wtime();
    myapp.open("tempi_esecuzione.txt",ios::out|ios::app);
    myapp<<endwtime-startwtime<<" ";
    myapp.close();

    startwtime = omp_get_wtime();
    ConvolutionZeroPaddingP(B,rows,columns,K);
    endwtime = omp_get_wtime();
    myapp.open("tempi_esecuzione.txt",ios::out|ios::app);
    myapp<<endwtime-startwtime<<" ";
    myapp.close();

    startwtime = omp_get_wtime();
    ReLU6P(B,rows,columns);
    endwtime = omp_get_wtime();
    myapp.open("tempi_esecuzione.txt",ios::out|ios::app);
    myapp<<endwtime-startwtime<<" "<<THREADS;
    myapp.close();
*/
    return 0;
}
