#include<fstream>
typedef float** matrix;

static matrix alloc_matrix (int  d1,int  d2);
matrix read_matrix (int & d1,int & d2);
matrix read_matrix_file (std::ifstream& infile, const int d1, const int d2);
void print_matrix (matrix a,int d1,int d3);
void print_matrix_file (std::ofstream& outfile, matrix a,int d1,int d3);


matrix MaxPool(matrix m,int &r,int &c); //per comodità e per confrotnare i risultati ho tenuto la versine sequenziale
matrix MaxPoolP(matrix m,int &r,int &c); //versione parallelizzata (P)
matrix AvgPool(matrix m,int &r,int &c);
matrix AvgPoolP(matrix m,int &r,int &c);
void ReLU6(matrix m,int &r,int &c);
void ReLU6P(matrix m,int &r,int &c);
matrix Convolution(matrix m,int &r,int &c, matrix ker);
matrix ConvolutionP(matrix m,int &r,int &c, matrix ker);
matrix ConvolutionZeroPadding(matrix m,int &r,int &c, matrix ker);
matrix ConvolutionZeroPaddingP(matrix m,int &r,int &c, matrix ker);


void generate(int dim); //per generare matrice e kernel usati per testare i tempi di esecuzione