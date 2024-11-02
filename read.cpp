//example on how to read matrix from the file

#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

typedef float** matrix;

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
    cout << "prima dimensione: ";
    cin >> d1;
    cout << "seconda dimensione: ";
    cin >> d2;
    res = alloc_matrix(d1,d2); //chiamo funzione ausiliaria

    for (i=0;i<d1;i++) {
        for (j=0;j<d2;j++) {
            cout << "[" << i << "," << j << "] : ";
            cin >> res[i][j];
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

    cout << endl;
    for (i=0;i<d1;i++){
        for (j=0;j<d3;j++)
            cout << a[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}


int main(int argc, char * argv[])  {
    int rows, columns;
    rows = 10; //just to test
    columns = 10; //just to test
    matrix  A;
    ifstream myin;
    char c;
    if (argc!=2) {
        cout << "Usage: ./a.out <targetfile>\n";
        exit(0);
    }
    myin.open(argv[1],ios::in);

    A = read_matrix_file(myin, rows, columns); //just as an example the limit is 10x10

    myin.close();

    print_matrix(A, rows, columns);

    return 0;
}
