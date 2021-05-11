using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixCHMLA
{
    public class Matrix
    {

        private double[,] data;

        public int n { get; }
        public int m { get; }

        public Matrix(int n, int m)
        {
            this.m = m;
            this.n = n;
            this.data = new double[n, m];
        }

        double this[int i, int j]
        {
            get
            {
                return this.data[i, j];
            }
            set
            {
                this.data[i, j] = value;
            }
        }
        //create matrix
        static Matrix CreateOwnMatrix(int size)
        {
            Matrix matrixx = new Matrix(size, size);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {

                    Console.Write($"matrix A[" + i + ',' + j + "]= ");
                    matrixx[i, j] = Convert.ToInt32(Console.ReadLine());
                }
            }

            return matrixx;
        }
         static Matrix CreateZeroMatrix(int size)
        {
            Matrix matrix = new Matrix(size, size);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = 0;
                }
            }
            return matrix;
        }
         static Matrix CreateOneDiahonalMatrix(int size)
        {
            Matrix matrix = new Matrix(size, size);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i == j)
                        matrix[i, j] = 1;
                    else
                        matrix[i, j] = 0;
                }
            }
            return matrix;
        }
         static Matrix CreateZeroVector(int size)
        {
            Matrix vector = new Matrix(size,1);
            for (int i = 0; i < size; i++)
                vector[i,0] = 0;
            return vector;
        }
         static Matrix CreateOwnVector(int size)
        {
            Matrix vector = new Matrix(size,1);
            for (int i = 0; i < size; i++)
            {
                Console.Write($"vector B[" + i + "]= ");
                vector[i,0] = Convert.ToInt32(Console.ReadLine());
            }
            return vector;
        }
         static Matrix FillMatrixGause(int size)
        {
            Matrix data = new Matrix(size, size + 1);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size+1; j++)
                {
                    if(j==size)
                        Console.Write($"Input b[{i}][{j}]: ");
                    else
                        Console.Write($"Input m[{i}][{j}]: ");
                    data[i, j] = Convert.ToInt32(Console.ReadLine());
                }
            }
            return data;
        }

        //printMatrix
         static void PrintMatrix(Matrix matrix, int size)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                    Console.Write(matrix[i, j] + " ");
                Console.WriteLine();
            }
        }
    
        static void PrintVector(Matrix vector, int size)
        {
            for (int i = 0; i < size; i++)
            {
                Console.Write($" {vector[i,0]}");
            }
        }


        //UL Metod to solve
         static Matrix FillMatrixULMetod(Matrix matrixA, int n, string str)
        {
            Matrix matrixL = new Matrix(n, n);
            matrixL = CreateZeroMatrix(n);
            Matrix matrixU = new Matrix(n, n);
            matrixU = CreateOneDiahonalMatrix(n);
            for (int i = 0; i < n; i++)
            {

                for (int j = 0; j < i + 1; j++)
                {
                    double sumaL = 0;
                    for (int k = 0; k <= j - 1; k++)
                    {
                        sumaL += matrixL[i, k] * matrixU[k, j];
                    }
                    matrixL[i, j] = matrixA[i, j] - sumaL;

                }

                if (matrixU[i, i] == 0)
                    throw new Exception("Error 404!\n\tMinor =0!");
                if (matrixL[i, i] == 0)
                    throw new Exception("Error 404!\n\tMinor =0!");
                for (int j = i + 1; j < n; j++)
                {
                    double sumaU = 0;
                    for (int k = 0; k <= j - 1; k++)
                    {
                        sumaU += matrixL[i, k] * matrixU[k, j];
                    }
                    matrixU[i, j] = (matrixA[i, j] - sumaU) / matrixL[i, i];

                }

            }
            switch (str)
            {
                case "L": return matrixL;
                case "U": return matrixU;

            }
            return null;
        }

        //find result Vector Y in UL metod
         static Matrix VectorYLU(Matrix vectorB, Matrix matrixL, int n)
        {
            Matrix vectorY = new Matrix(n, 1);
            vectorY = CreateZeroMatrix(n);
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int k = 0; k <= i; k++)
                {
                    sum += (matrixL[i, k] * vectorY[k, 0]);
                }
                vectorY[i, 0] = (vectorB[i, 0] - sum) / matrixL[i, i];
                sum = 0;
            }
            return vectorY;
        }
        //find result Vector X in UL metod RESULT
         static Matrix VectorXLU(Matrix vectorY, Matrix matrixU, int n)
        {
            Matrix vectorX = new Matrix(n, 1);
            vectorX = CreateZeroMatrix(n);
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0;
                for (int k = i + 1; k < n; k++)
                {
                    sum += (matrixU[i, k] * vectorX[k, 0]);
                }
                vectorX[i, 0] = vectorY[i, 0] - sum;
            }
            return vectorX;
        }
        public static Matrix operator *(Matrix matrix, Matrix matrix2)
        {
           
            Matrix multiplicatedMatrix = new Matrix(matrix.m, matrix2.n);
            double temp;

            for (int i = 0; i < multiplicatedMatrix.n; i++)
            {
                for (int j = 0; j < multiplicatedMatrix.m; j++)
                {
                    temp = 0;
                    for (int k = 0; k < matrix.n; k++)
                    {
                        temp += matrix[i, k] * matrix2[k, j];

                    }
                    multiplicatedMatrix[i, j] = temp;
                }
            }
            return multiplicatedMatrix;
        }
         static Matrix MultiplicateMatrixToVector(Matrix matrix1, Matrix matrix2, int size)
        {
            Matrix multiplicatedMatrix = new Matrix(size,1);
            double temp = 0;
            for (int i = 0; i < size; i++)
            {
                temp = 0;
                for (int j = 0; j < size; j++)
                {
                    temp += matrix1[i, j] * matrix2[j, 0];

                    // multiplicatedMatrix[i] = temp;
                }
                multiplicatedMatrix[i,0] = temp;
            }

            return multiplicatedMatrix;
        }
        //check if matrix A = multiplicated U*L matrix
        static bool CheckMatrix_LU(Matrix matrixA, Matrix matrixATemp, int size)
        {
            bool checking = true;

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (matrixA[i, j] != matrixATemp[i, j])
                    {
                        checking = false;
                        break;
                    }


                }
            }
            return checking;
        }
        //determinant matrix 
         static double DeterminantMatrix_LUMetod(Matrix matrixL, int size)
        {
            double det = 1;

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i == j)
                    {
                        det *= matrixL[i, j];

                    }
                }
            }

            return det;
        }
        //check if A*x = b vector
         static bool CheckResult_LU(Matrix matrixxTemp, Matrix x, Matrix b, int n)
        {
            int i, j;
            double summa;
            bool result = true;
            for (i = 0; i < n; ++i)
            {
                summa = 0;
                for (j = 0; j < n; ++j)
                {
                    summa += (matrixxTemp[i, j] * x[j,0]);

                }

                if (summa != b[i,0])
                {
                    result = false;
                    break;
                }
                else
                    result = true;


            }
            return result;
        }


         void ProcessFunctionOutput(Action<int, int> func)
        {
            for (var i = 0; i < this.m; i++)
            {
                for (var j = 0; j < this.n; j++)
                {
                    func(i, j);
                }
                Console.WriteLine("");
            }
        }

         void ProcessFunctionOverData(Action<int, int> func)
         {
            for (var i = 0; i < this.m; i++)
            {
                for (var j = 0; j < this.n; j++)
                {
                    func(i, j);
                }

            }
         }
         void SwapRow(int k, int l)
         {
            for (var i = 0; i < this.n; i++)
            {
                double t = data[k, i];
                data[k, i] = data[l, i];
                data[l, i] = t;
            }
         }

         int MaxElem(int index)
         {
            int FinalIndex = index;
            double MaxElem = data[index, index];
            for (var i = index + 1; i < this.n; i++)
            {
                if (Math.Abs(data[i, index]) > Math.Abs(MaxElem))
                {
                    MaxElem = data[i, index];
                    FinalIndex = i;
                }

            }
            return FinalIndex;
         }

         void SingleMatrix()
         {
            ProcessFunctionOverData((i, j) =>
            {
                if (i == j)
                    data[i, j] = 1;
                else
                    data[i, j] = 0;
            });
         }

        
         void OutputMatrix()
        {
            ProcessFunctionOutput((i, j) => {
                Console.Write(data[i, j] + " ");
            });
        }

         void Round()
        {
            ProcessFunctionOutput((i, j) => {
                data[i, j] = Math.Round(data[i, j], 3);
                Console.Write(data[i, j] + " ");
            });
        }

         void OutPutMatrixToInt()
        {
            ProcessFunctionOutput((i, j) => {
                Console.Write(Math.Round(Math.Abs(data[i, j])) + " ");
            });

        }

         void CopyMatrix(Matrix matrix)
        {
            ProcessFunctionOverData((i, j) =>
            {
                data[i, j] = matrix[i, j];
            });
        }

     


         Matrix CreateMatrixWithoutRow(int row)
        {
            if (row < 0 || row >= this.m)
            {
                throw new ArgumentException("invalid row index");
            }
            Matrix result = new Matrix(this.m - 1, this.n);
            result.ProcessFunctionOverData((i, j) => result[i, j] = i < row ? this[i, j] : this[i + 1, j]);
            return result;
        }

         Matrix CreateMatrixWithoutColumn(int column)
        {
            if (column < 0 || column >= this.n)
            {
                throw new ArgumentException("invalid column index");
            }
            var result = new Matrix(this.n, this.m - 1);
            result.ProcessFunctionOverData((i, j) => result[i, j] = j < column ? this[i, j] : this[i, j + 1]);
            return result;
        }


         double CalculateDeterminant()
        {

            if (this.n == 2)
            {
                return this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0];
            }
            double result = 0;
            for (var j = 0; j < this.n; j++)
            {
                result += (j % 2 == 1 ? 1 : -1) * this[1, j] * this.CreateMatrixWithoutColumn(j).CreateMatrixWithoutRow(1).CalculateDeterminant();
            }
            return result;
        }

         bool IsSymmetric()
        {
            bool IsTrue = true;
            ProcessFunctionOverData((i, j) => {
                if (!(data[i, j] == data[j, i]))
                    IsTrue = false;
            });
            return IsTrue;
        }

         Matrix Transpose()
        {
            Matrix result = new Matrix(n, n);
            ProcessFunctionOverData((i, j) => {
                result[j, i] = data[i, j];
            });
            return result;
        }

        static Matrix UU(Matrix A, Matrix U, int n, Matrix b)
        {
            double[] y = new double[n];
            Matrix x = new Matrix(n, 1);
            for (int i = 0; i < n; i++)
            {
                double SumK = 0;
                for (int k = 0; k <= i - 1; k++)
                {
                    SumK += (U[k, i] * U[k, i]);
                }
                if (SumK > A[i, i])
                    throw new Exception("Division by 0");

                U[i, i] = Math.Sqrt(A[i, i] - SumK);

                if (U[i, i] == 0)
                    throw new Exception("Diagonal element is 0");
                for (int j = i + 1; j < n; j++)
                {
                    SumK = 0;
                    for (int k = 0; k <= i - 1; k++)
                    {
                        SumK += (U[k, i] * U[k, j]);
                    }
                    U[i, j] = (A[i, j] - SumK) / U[i, i];
                }

            }
            for (int i = 0; i < n; i++)
            {
                double SumK = 0;
                for (int k = 0; k <= i - 1; k++)
                {
                    SumK += (U[k, i] * y[k]);
                }
                y[i] = (1 / U[i, i]) * (b[i, 0] - SumK);

            }
            for (int i = n - 1; i >= 0; i--)
            {
                double SumK = 0;
                for (int k = i + 1; k < n; k++)
                {
                    SumK += (U[i, k] * x[k, 0]);
                }
                x[i, 0] = (1 / U[i, i]) * (y[i] - SumK);
            }

            return x;

        }


         static Matrix TMA(Matrix A, Matrix B, Matrix C, Matrix F, int n)
        {
            n++;
            Matrix y = new Matrix(n, 1);
            double[] ksi = new double[n - 1];
            double[] eta = new double[n - 1];


            ksi[n - 2] = (-1) * (A[n - 2, 0]) / C[n - 1, 0];
            eta[n - 2] = F[n - 1, 0] / C[n - 1, 0];
            for (int i = n - 2; i > 0; i--)
            {
                double id = (C[i, 0] + (B[i, 0] * ksi[i]));
                ksi[i - 1] = ((-1) * A[i - 1, 0]) / id;
                eta[i - 1] = (F[i, 0] - (B[i, 0] * eta[i])) / id;
            }


            y[0, 0] = (F[0, 0] - (B[0, 0] * eta[1])) / (C[0, 0] + (B[0, 0] * ksi[1]));
            for (int i = 1; i < n; i++)
                y[i, 0] = ksi[i - 1] * y[i - 1, 0] + eta[i - 1];

            return y;
        }

         static void FillAllVector(Matrix A, Matrix B, Matrix C, Matrix F, int n)
        { 
            double h = 1.0 / (n);

            A[n - 1, 0] = 0;
            B[0, 0] = 0;
            C[0, 0] = 1;
            C[n, 0] = 1;
            F[0, 0] = 1;
            F[n, 0] = 3;
            for (int i = 0; i < n; i++)
            {
                if (i <= n - 2)
                {
                    A[i, 0] = 1;
                }
                if (i >= 1 && i <= n - 1)
                {
                    B[i, 0] = 1;
                }
                if (i >= 1 && i <= n - 1)
                {
                    F[i, 0] = ((4 - (1 + i * h) * (2 * Math.Pow(i, 2) * Math.Pow(h, 2) + 1)) * Math.Pow(h, 2));
                    C[i, 0] = (-2 - (1 + i * h) * Math.Pow(h, 2));
                }

            }

        }

         static void Norma(Matrix y, int n)
        {
            double h = 1.0 / (n);


            double[] tempY = new double[n + 1];
            for (int i = 0; i <= n; i++)
            {
                tempY[i] = (((2 * Math.Pow(i, 2) * Math.Pow(h, 2)) + 1));
            }

            double[] resultNorma = new double[n + 1];
            for (int i = 0; i <= n; i++)
            {
                resultNorma[i] = (Math.Abs(y[i, 0] - tempY[i]));
            }

            Console.WriteLine($"Norma: ||y-y*|| = {resultNorma.Max()}");
        }
        
         static void StabilityMatrix(Matrix A, Matrix B, Matrix C, int n)
        {

            int count = 0;
            if ((Math.Abs(C[0,0]) > Math.Abs(B[0,0])))
            {
                count++;
            }
            else if (!(Math.Abs(C[0,0]) == Math.Abs(B[0,0])))
            {
                throw new Exception("Not correct task!");
            }
            for (int i = 0; i < n; i++)
            {
                if (!(Math.Abs(C[i,0]) > 0))
                {
                    throw new Exception("Not correct task!");
                }
                if (i < n - 1)
                {
                    if (!(Math.Abs(A[i,0]) >= 0))
                    {
                        throw new Exception("Not correct task!");
                    }
                    if (!(Math.Abs(B[i,0]) >= 0))
                    {
                        throw new Exception("Not correct task!");
                    }
                }
                if (i >= 1 && i < n - 1)
                {
                    if ((Math.Abs(C[i,0]) > (Math.Abs(A[i,0]) + Math.Abs(B[i,0]))))
                    {
                        count++;

                    }
                    else if (!(Math.Abs(C[i,0]) == (Math.Abs(A[i,0]) + Math.Abs(B[i,0]))))
                    {
                        throw new Exception("Not correct task!");
                    }
                }

            }
            if ((Math.Abs(C[n - 1, 0]) > Math.Abs(B[n - 2, 0])))
            {
                count++;

            }
            else if (!(Math.Abs(C[n - 1, 0]) == Math.Abs(B[n - 2, 0])))
            {
                throw new Exception("Not correct task!");
            }
            if (!(count > 0))
            {
                throw new Exception("Not correct task!");
            }
            Console.WriteLine("Correct task based on the theorem of Stability!");
        }


        static void Jacobi(int N, Matrix A, Matrix F, Matrix X, int countInterationForCycle)
        {

            double eps, norma;
            Console.Write("\nInput value of eps to have better answers:");
            eps = Convert.ToDouble(Console.ReadLine());

            double[] TempX = new double[N];

            for (int k = 0; k < N; k++)
                TempX[k] = X[k,0];
            int countIter = 0;

            do
            {
                for (int i = 0; i < N; i++)
                {
                    TempX[i] = F[i,0];
                    for (int g = 0; g < N; g++)
                        if (i != g)
                            TempX[i] -= A[i, g] * X[g,0];
                    TempX[i] /= A[i, i];
                }
                norma = Math.Abs(X[0,0] - TempX[0]);
                for (int h = 0; h < N; h++)
                {
                    if (Math.Abs(X[h,0] - TempX[h]) > norma)
                        norma = Math.Abs(X[h,0] - TempX[h]);
                    X[h,0] = TempX[h];
                }
                countIter++;
            }
            while (norma > eps && countInterationForCycle > countIter);



            //    Console.WriteLine($"\n||A*x^(k+1) - b||= {norma}\n");

            Console.WriteLine("Count of iteration: " + countIter);

        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///             METODS //////////////////////////////
        ///            

        public static void Gauus_Metod()
        {
            Console.Write("Input size of matrix(matrix is with vector b): ");
            int n = Convert.ToInt32(Console.ReadLine());
            Matrix matrix = Matrix.FillMatrixGause(n);

            Matrix.PrintMatrix(matrix,n);
            for (int k = 0; k < n - 1; k++)
            {
                int row_index_with_max_elem = matrix.MaxElem(k);
                matrix.SwapRow(k, row_index_with_max_elem);
                for (int i = k + 1; i < n; i++)
                {
                    double m = (-1) * (matrix[i, k]) / (matrix[k, k]);
                    for (int j = k; j < n + 1; j++)
                    {
                        matrix[i, j] = matrix[i, j] + m * matrix[k, j];
                    }
                }
            }


            Matrix x = new Matrix(n,1);
           
            for (int k = n - 1; k >= 0; k--)
            {
                double s = 0;
                for (int j = k; j < n; j++)
                {
                    s += matrix[k, j] * x[j,0];
                }

                x[k,0] = (matrix[k, n] - s) / matrix[k, k];
            }


            Console.WriteLine("\n");

            Matrix.PrintMatrix(matrix, n);

            Console.WriteLine("\n");

            for (int i = 0; i < n; i++)
            {
                Console.WriteLine($"X:{x[i,0]}");
            }

            double matrixDet = 1;
            for (int i = 0; i < matrix.n; i++)
            {
                for (int j = 0; j < matrix.m; j++)
                {
                    if (i == j)
                        matrixDet *= matrix[i, j];
                }
            }
            Console.WriteLine($"\nMatrix det:{matrixDet}");
            Console.ReadKey();
            Console.Clear();
        }

        public static void LU_2Metod()
        {
            int n = 0;
            Console.Write("Input size of matrix: ");
            n = Convert.ToInt32(Console.ReadLine());

            //general matrix
            Matrix matrixA = Matrix.CreateOwnMatrix(n);
            Console.WriteLine();
            //vector b
            Matrix vectorB = Matrix.CreateOwnVector(n);

            Matrix matrixL = new Matrix(n, n);
            matrixL = Matrix.FillMatrixULMetod(matrixA, n, "L");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("\n\n\n");
            Matrix matrixU = new Matrix(n, n);
            matrixU = Matrix.FillMatrixULMetod(matrixA, n, "U");

            Console.WriteLine("matrix A:");
            Matrix.PrintMatrix(matrixA, n);
            Console.WriteLine();

            Console.WriteLine("matrix L:");
            Matrix.PrintMatrix(matrixL, n);
            Console.WriteLine("matrix U:");
            Matrix.PrintMatrix(matrixU, n);


            Matrix ULMatrix = matrixL* matrixU;

            Console.WriteLine($"\nCheking if matrix A = matrix LU: {Matrix.CheckMatrix_LU(matrixA, ULMatrix, n)}");

            double det = Matrix.DeterminantMatrix_LUMetod(matrixL, n);
            Console.WriteLine($"\n\tDeterminant of matrix A: {det}");

            Matrix vectorY = new Matrix(n,1);
            vectorY = Matrix.VectorYLU(vectorB, matrixL, n);
            Console.Write("Y vector: ");
            Matrix.PrintVector(vectorY, n);

            Console.WriteLine();

            Matrix vectorX = new Matrix(n,1);
            vectorX = Matrix.VectorXLU(vectorY, matrixU, n);
            Console.Write("X vector: ");
            Matrix.PrintVector(vectorX, n);

            Console.WriteLine($"\nCheck result: {Matrix.CheckResult_LU(matrixA, vectorX, vectorB, n)}");

            Console.ReadKey();
            Console.Clear();
        }

        public static void UtU_Metod()
        {
            Console.WriteLine("Input n: ");
            int n = Int32.Parse(Console.ReadLine());
            Matrix A = CreateOwnMatrix(n);
            Console.WriteLine("\nEnter A Elem:");
            
            if (A.IsSymmetric())
            {
                Matrix U = new Matrix(n, n);
                Matrix b = CreateOwnVector(n);
                Console.WriteLine("\nEnter b Elem:");
                
                 Matrix x = new Matrix(n, 1);

                x = UU(A, U, n, b);

                Console.WriteLine("\nx :");
                PrintVector(x, n);

                Console.WriteLine("\nU :");
                PrintMatrix(U,n);

                double detA = 1;
                for (int i = 0; i < n; i++)
                {
                    detA *= U[i, i];
                }
                Console.WriteLine($"\ndet(A) = det(U)^2 = {Math.Pow(detA, 2.0F)}");

                Console.WriteLine("\nU^t * U :");
                Matrix tmpU = U.Transpose() * U;
                PrintMatrix(tmpU, n);

                Console.WriteLine("\nA :");
                PrintMatrix(A,n);

                Console.WriteLine("\nA * x :");
                Matrix tmp = A * x;
                PrintVector(tmp, n);

                Console.WriteLine("\nb :");
                PrintVector(b, n);

                Console.ReadKey();
                
            }
            else
            {
                throw new Exception("Matrix not symmetric");
            }


            Console.ReadKey();
            Console.Clear();
        }
        public static void LeftRunMatrixAndStability()
        {

            Console.Write("Input n: ");
            int n = Convert.ToInt32(Console.ReadLine());

            Matrix A = new Matrix(n,1);
            Matrix B = new Matrix(n, 1);
            Matrix C = new Matrix(n+1, 1);
            Matrix F = new Matrix(n+1, 1);
            Matrix.FillAllVector(A, B, C, F, n);

            Matrix.StabilityMatrix(A, B, C, n);

            Console.Write("\n\nA :");
            Matrix.PrintVector(A, n);
            Console.Write("\nB :");
            Matrix.PrintVector(B, n);
            Console.Write("\nC :");
            Matrix.PrintVector(C, n + 1);
            Console.Write("\nF :");
            Matrix.PrintVector(F, n + 1);
            Console.WriteLine();


            Matrix D = new Matrix(n + 1, n + 1);
            for (int i = 0; i < n + 1; i++)
            {
                D[i, i] = C[i, 0];
                if (i < n)
                {
                    D[i, i + 1] = B[i, 0];
                    D[i + 1, i] = A[i, 0];
                }
            }
             
            Matrix y = Matrix.TMA(A, B, C, F, n);
           
            Console.Write("\n\nY :");
            Matrix.PrintVector(y, n + 1);

            Console.WriteLine();
            Matrix.Norma(y, n);
            Console.WriteLine();

            Console.Write("\nD*y :");
            Matrix.PrintVector(Matrix.MultiplicateMatrixToVector(D,y, n+1), n + 1);
            Console.WriteLine();

        }

        static public void Realization_JacobiMethod()
        {
            Console.Write("Input size of matrix:");
            int n = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine("\nMatrix A:");
            Matrix myMatrix = Matrix.CreateOwnMatrix(n);
            Console.WriteLine("\nb:  ");
            Matrix bVector = Matrix.CreateOwnVector(n);
            Matrix yVector = new Matrix(n,1);
            Matrix xVector = new Matrix(n,1);

            Console.Write("\nCount of iteration of matrix:");
            int iteration = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine("\nMatrix:");
            Matrix.PrintMatrix(myMatrix, n);
            Console.Write("\nb:  ");
            Matrix.PrintVector(bVector, n);


            for (int i = 0; i < n; i++)
                xVector[i,0] = 1.0;

            Matrix.Jacobi(n, myMatrix, bVector, xVector, iteration);

            Matrix Ax = MultiplicateMatrixToVector(myMatrix ,xVector,n);
            double Norma_AxB = 0;
            for (int i = 0; i < n; i++)
            {
                Norma_AxB += (Math.Abs(Ax[i,0] - bVector[i,0]));
            }
            Console.WriteLine($"\n||A*x^(k+1) - b||= {Norma_AxB}\n");

            Console.Write("\nResult x:  ");
            Matrix.PrintVector(xVector, n);
            Console.ReadKey();
        }
    }
}

