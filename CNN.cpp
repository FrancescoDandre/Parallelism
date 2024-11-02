#include "CNN.h"
#include<iostream>
#include<fstream>
#include<omp.h>
#include<ctime>
using namespace std;

//funzioni per allocare la matrice,leggerla e stamparla sia attraverso standardIO che attraverso file
static matrix alloc_matrix (int  d1,int  d2)
{
    int i,j;

    matrix res;
    res = new float *[d1];
    for (i=0;i<d1;i++) {
        res[i] = new float[d2];
    }
    return res;
}

matrix read_matrix (int & d1,int & d2)
{
    int i,j;

    matrix res;
    std::cout << "prima dimensione: ";
    std::cin >> d1;
    std::cout << "seconda dimensione: ";
    std::cin >> d2;
    res = alloc_matrix(d1,d2); //chiamo funzione ausiliaria

    for (i=0;i<d1;i++) {
        for (j=0;j<d2;j++) {
            std::cout << "[" << i << "," << j << "] : ";
            std::cin >> res[i][j];
        }
    }
    return res;
}

matrix read_matrix_file (ifstream& infile, const int d1, const int d2)
{
    int i,j;
    matrix res;
    res = alloc_matrix(d1,d2);

    for (i=0;i<d1;i++) {
        for (j=0;j<d2;j++) {
            infile >> res[i][j];
        }
    }
    return res;
}

void print_matrix (matrix a,int d1,int d3)
{
    int i,j;

    std::cout << endl;
    for (i=0;i<d1;i++){
        for (j=0;j<d3;j++)
            std::cout << a[i][j] << " ";
        std::cout << endl;
    }
    std::cout << endl;
}

void print_matrix_file (ofstream& outfile, matrix a,int d1,int d3)
{
    int i,j;
    for (i=0;i<d1;i++){
        for (j=0;j<d3;j++)
            outfile << a[i][j] << " ";
        outfile << endl;
    }
    outfile << endl;
}

// Operazioni in CNN

matrix MaxPool(matrix m,int &r,int &c){
    matrix new_m = alloc_matrix(r/4,c/4);
    int i,j,h,k;
    for (i=0;i<r/4;i++){
        for (j=0;j<c/4;j++){
            new_m[i][j]=m[4*i][4*j];
        }
    }
    for(h=0;h<r/4;h++){
        for(k=0;k<c/4;k++){
            for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                    if(m[i+4*h][j+4*k]>new_m[h][k]) new_m[h][k]=m[i+4*h][j+4*k];
                }
            }
        }
    }
    return new_m;
}

matrix MaxPoolP(matrix m,int &r,int &c){
    matrix new_m = alloc_matrix(r/4,c/4);
    int i,j,h,k;
    #pragma omp parallel shared(r,c,m,new_m) private(i,j,h,k) default(none)
    {
        #pragma omp for
        for (i=0;i<r/4;i++){
            for (j=0;j<c/4;j++){
                new_m[i][j]=m[4*i][4*j];
            }
        }
     
        #pragma omp for
        for(h=0;h<r/4;h++){
            for(k=0;k<c/4;k++){
                for (i=0;i<4;i++){
                    for (j=0;j<4;j++){
                        if(m[i+4*h][j+4*k]>new_m[h][k]) new_m[h][k]=m[i+4*h][j+4*k];
                    }
                }
            }
        }
    }
    return new_m;
}

matrix AvgPool(matrix m,int &r,int& c){
    matrix new_m = alloc_matrix(r/4,c/4);
    int i,j,h,k;
    for (i=0;i<r/4;i++){
        for (j=0;j<c/4;j++){
            new_m[i][j]=0;
        }
    }
    for(h=0;h<r/4;h++){
        for(k=0;k<c/4;k++){
            for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                    new_m[h][k]+=m[i+4*h][j+4*k];
                }
            }
            new_m[h][k]/=16;
        }
    }
    return new_m;
}

matrix AvgPoolP(matrix m,int &r,int& c){
    matrix new_m = alloc_matrix(r/4,c/4);
    //float count=0;
    #pragma omp parallel shared(r,c,m,new_m/*,count*/) default(none)
    {
        int i,j,h,k;
        #pragma omp for        
        for (i=0;i<r/4;i++){
            for (j=0;j<c/4;j++){
                new_m[i][j]=0;
            }
        }

        #pragma omp for //reduction(+:count)
        for(h=0;h<r/4;h++){
            for(k=0;k<c/4;k++){
                for (i=0;i<4;i++){
                    for (j=0;j<4;j++){
                        new_m[h][k]+=m[i+4*h][j+4*k];
                        //count+=m[i+4*h][j+4*k];
                        //new_m[h][k]=count;
                    }
                }
                //count=0;
                new_m[h][k]/=16;
            }
        }
    }
    return new_m;
}

void ReLU6(matrix m,int &r,int &c){
    int i,j,h,k;
    for (i=0;i<r;i++){
        for (j=0;j<c;j++){
            if(m[i][j]<0) m[i][j]=0;
            else if(m[i][j]>6) m[i][j]=6;
        }
    }
}

void ReLU6P(matrix m,int &r,int &c){
    int i,j,h,k;
    #pragma omp parallel for shared(r,c,m) private(i,j,h,k) default(none)
    for (i=0;i<r;i++){
        for (j=0;j<c;j++){
            if(m[i][j]<0) m[i][j]=0;
            else if(m[i][j]>6) m[i][j]=6;
        }
    }
}

matrix Convolution(matrix m, int &r,int &c, matrix ker){
    matrix new_m = alloc_matrix(r-3,c-3);
    int i,j,h,k;
    for (i=0;i<r-3;i++){
        for (j=0;j<c-3;j++){
            new_m[i][j]=0;
        }
    }
    for(h=0;h<r-3;h++){
        for(k=0;k<c-3;k++){
            for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                    new_m[h][k]+=m[i+h][j+k]*ker[i][j];
                }
            }
        }
    }
    return new_m;
}

matrix ConvolutionP(matrix m, int &r,int &c, matrix ker){
    matrix new_m = alloc_matrix(r-3,c-3);
    int i,j,h,k;
    //float count=0;
    #pragma omp parallel shared(m,r,c,ker,new_m/*,count*/) private(i,j,h,k) default(none)
    {    
        #pragma omp for
        for (i=0;i<r-3;i++){
            for (j=0;j<c-3;j++){
                new_m[i][j]=0;
            }
        }
        #pragma omp for //reduction(+:count)
        for(h=0;h<r-3;h++){
            for(k=0;k<c-3;k++){
                for (i=0;i<4;i++){
                    for (j=0;j<4;j++){
                        new_m[h][k]+=m[i+h][j+k]*ker[i][j];
                        //count+=m[i+h][j+k]*ker[i][j];
                        //new_m[h][k]=count;
                    }
                }
                //count=0;
            }
        }
    }
    return new_m;
}

matrix ConvolutionZeroPadding(matrix m, int &r,int &c, matrix ker){
    matrix new_m = alloc_matrix(r,c);
    int i,j,h,k;
    for (i=0;i<r;i++){
        for (j=0;j<c;j++){
            new_m[i][j]=0;
        }
    }
    for(h=0;h<r;h++){
        for(k=0;k<c;k++){
            for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                    if(i+h>r-1 || j+k>c-1) break;
                    new_m[h][k]+=m[i+h][j+k]*ker[i][j];
               }
            }
        }
    }
    return new_m;
}

matrix ConvolutionZeroPaddingP(matrix m, int &r,int &c, matrix ker){
    matrix new_m = alloc_matrix(r,c);
    int i,j,h,k;
    //float count=0;  //per opzione con reduction
    #pragma omp parallel shared(m,r,c,ker,new_m/*,count*/) private(i,j,h,k) default(none) 
    {    
        #pragma omp for 
        for (i=0;i<r;i++){
            for (j=0;j<c;j++){
                new_m[i][j]=0;
            }
        }
        #pragma omp for //reduction(+:count)   //collapse(4)
        for(h=0;h<r;h++){
            for(k=0;k<c;k++){
                for (i=0;i<4;i++){
                    for (j=0;j<4;j++){
                        if(i+h>r-1 || j+k>c-1) break;
                        new_m[h][k]+=m[i+h][j+k]*ker[i][j];
                        //count+=m[i+h][j+k]*ker[i][j]; //tentativo di parallelizzare usando reduction
                        //new_m[h][k]=count;
                    }
                }
                //count=0;
            }
        }
    }
    return new_m;
}

void generate(int dim)  {

    float LO; //lower bound for random values
    float HI; //upper bound for random values

    fstream outM, outK;

    outM.open("matrice_di_test.txt",ios::out);

    //generate input "image"
    srand((int)time(NULL));
    LO = -10.0;
    HI = 10.0;
    for (int i=0;i<dim;i++) {
        for (int j = 0; j < dim; j++) {
            outM << LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))) << " ";
        }
        outM << '\n';
    }

    outM.close();

    //generate kernel (filter)
    outK.open("kernel_di_test.txt",ios::out);
    srand((int)time(NULL));
    LO = -1.0;
    HI = 1.0;
    for (int i=0;i<4;i++) {
        for (int j = 0; j < 4; j++) {
            outK << LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))) << " ";
        }
        outK << '\n';
    }
    outK.close();
}