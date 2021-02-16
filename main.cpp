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

    double localIntg = Trap(local_a, local_b, local_n, h, myFun);
    if (myRank == 0)
    {
        double totalIng = localIntg;
        for (int src = 1; src < worldSize; src++)
        {
            MPI_Recv(&localIntg, 1, MPI_DOUBLE, src, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            totalIng += localIntg;
        }
        cout << "Int(f(x), x = " << a << ".." << b << ") = ";
        cout.precision(16);
        cout << totalIng << endl;
    }
    else
    {
        MPI_Send(&localIntg, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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

        for (int dest = 1; dest < size; dest++)
        {
            tag = 0;
            MPI_Send(&fun, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            tag = 1;
            MPI_Send(&a, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            tag = 2;
            MPI_Send(&b, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            tag = 3;
            MPI_Send(&n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        tag = 0;
        MPI_Recv(&fun, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        tag = 1;
        MPI_Recv(&a, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        tag = 2;
        MPI_Recv(&b, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        tag = 3;
        MPI_Recv(&n, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    }
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