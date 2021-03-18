namespace MulticlassesDemo
{
    public static class MakeMatrikForNeuralNetwork
    {
        public static double[][] CreateMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];

            for (int r = 0; r < result.Length; ++r)
            {
                result[r] = new double[cols];
            }

            return result;
        }
    }
}
