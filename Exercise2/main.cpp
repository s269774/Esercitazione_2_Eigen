#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;


VectorXd SolvePALU(const MatrixXd& A,
                         const VectorXd& b)
{
    VectorXd solvePALU = A.fullPivLu().solve(b);
    return solvePALU;
}

VectorXd SolveQR(const MatrixXd& A,
                       const VectorXd& b)
{
    VectorXd solveQR = A.colPivHouseholderQr().solve(b);
    return solveQR;
}

void TestSolution(const MatrixXd& A,
                  const VectorXd& b,
                  const VectorXd& sol,
                  double& errRelPALU,
                  double& errRelQR)
{
    errRelPALU = (SolvePALU(A,b)-sol).norm()/sol.norm();
    errRelQR = (SolveQR(A,b)-sol).norm()/sol.norm();
}


int main()
{
    Vector2d sol(-1.0000e+0, -1.0000e+00);


    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1 = {-5.169911863249772e-01, 1.672384680188350e-01};

    double errRelPALU1, errRelQR1;
    TestSolution(A1, b1, sol, errRelPALU1, errRelQR1);
    cout << scientific << "1 - "<< "ErrPALU: "<< errRelPALU1<< " ErrQR: "<< errRelQR1<< endl;



    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2 = {-6.394645785530173e-04, 4.259549612877223e-04};

    double errRelPALU2, errRelQR2;
    TestSolution(A2, b2, sol, errRelPALU2, errRelQR2);
    cout<< scientific<< "2 - "<< "ErrPALU: "<< errRelPALU2<< " ErrQR: "<< errRelQR2<< endl;



    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3 = {-6.400391328043042e-10, 4.266924591433963e-10};

    double errRelPALU3, errRelQR3;
    TestSolution(A3, b3, sol, errRelPALU3, errRelQR3);
    cout<< scientific<< "3 - "<< "ErrPALU: "<< errRelPALU3<< " ErrQR: "<< errRelQR3<< endl;


    return 0;
}


