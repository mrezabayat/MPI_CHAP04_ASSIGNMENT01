#include "mpi.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

void GetData(int &fun, double &a, double &b, int &n, int myRank, int size);
double Trap(double local_a, double local_b, int local_n, double h, double (*pFunction)(double));
double x2(double x) { return x * x; }
// double sinx(double x) { return sin(x); }
// double expx(double x) { return exp(x); }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int myRank{}, worldSize{};
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    double a{}, b{};
    int n{}, fun{};
    GetData(fun, a, b, n, myRank, worldSize);
    double h = (b - a) / n;
    int local_n = n / worldSize;
    double local_a = a + myRank * local_n * h;
    double local_b = local_a + local_n * h;

    double (*myFun)(double);
    switch (fun)
    {
    case 1:
        myFun = &x2;
        break;
    case 2:
        myFun = &sin;
        break;
    case 3:
        myFun = &exp;
        break;
    default:
        cout << "Invalid choice for function!" << endl;
        break;
    }
    double localIntegral = Trap(local_a, local_b, local_n, h, myFun);
    double totalIntegral{};
    MPI_Reduce(&localIntegral, &totalIntegral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        cout << "Int(f(x), x = " << a << ".." << b << ") = ";
        cout.precision(16);
        cout << totalIntegral << endl;
    }
    
    MPI_Finalize();
    return 0;
}

void GetData(int &fun, double &a, double &b, int &n, int myRank, int size)
{
    int source = 0;
    int tag{};
    MPI_Status status;
    if (myRank == 0)
    {
        cout << "Choose a function to integrate: \n"
             << "1: f(x) = x^2 \n"
             << "2: f(x) = sin(x) \n"
             << "3: f(x) = exp(x)\n";
        cin >> fun;
        cout << "Enter a: ";
        cin >> a;
        cout << "Enter b: ";
        cin >> b;
        cout << "Enter n (note that it must be divisible by number of processes): ";
        cin >> n;
        assert(n % size == 0);
    }

    MPI_Bcast(&fun, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

double Trap(double local_a, double local_b, int local_n, double h, double (*pFunction)(double))
{
    double integral = (pFunction(local_a) + pFunction(local_b)) / 2;
    double x = local_a;
    for (int i = 1; i < local_n; i++)
    {
        x = x + h;
        integral = integral + pFunction(x);
    }
    integral = integral * h;
    return integral;
}