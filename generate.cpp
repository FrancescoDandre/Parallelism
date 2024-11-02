//This code can be used to generate random values matrices
//In the default configuration it generates (OVERWRITES!) a 128x128 float matrix.txt and a 4x4 kernel.txt

#include <iostream>
#include <fstream>
#include <cstdlib>
#include<ctime>
using namespace std;
int main(int argc, char * argv[])  {

    float LO; //lower bound for random values
    float HI; //upper bound for random values

    fstream outM, outK;
    char c;
    if (argc!=3) {
        cout << "Usage: ./a.out <targetMatrix> <targetKernel>\n";
        exit(0);
    }

    outM.open(argv[1],ios::out);

    //generate input "image"
    srand((int)time(NULL));
    LO = -10.0;
    HI = 10.0;
    for (int i=0;i<128;i++) {
        for (int j = 0; j < 128; j++) {
            outM << LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))) << " ";
        }
        outM << '\n';
    }

    outM.close();

    //generate kernel (filter)
    outK.open(argv[2],ios::out);
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
    return 0;
}