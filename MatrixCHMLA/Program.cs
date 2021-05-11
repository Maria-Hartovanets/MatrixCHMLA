using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixCHMLA
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\t\t\tMetod to solve matrix:");
            Console.WriteLine(" 1. Gauuse metod;\n" +
                " 2. LU metod the second way;\n" +
                " 3. U^t*U metod;\n" +
                " 4. Left runinng of the matrix wuth theorem of Stability;\n" +
                " 5. Jacobi Method\n");
            int op = Convert.ToInt32(Console.ReadLine());
            switch (op)
            {
                case 1:
                    Matrix.Gauus_Metod();
                    break;
                case 2:
                    Matrix.LU_2Metod();
                    break;
                case 3:
                    Matrix.UtU_Metod();
                    break;
                case 4:
                    Matrix.LeftRunMatrixAndStability();
                    break;
                case 5:
                    Matrix.Realization_JacobiMethod();
                    break;
            }
        }
    }
}
