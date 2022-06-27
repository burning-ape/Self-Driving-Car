using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System.IO;
using System.Linq;

public class SaveAndLoadSystem
{
    public static void SaveData(List<NeuralNetwork> populations)
    {
        Dictionary<int, DataToSave> dd = new Dictionary<int, DataToSave>();

        for (int i = 0; i < populations.Count; i++)
        {
            DataToSave dataT = new DataToSave()
            {
                _fitness = populations[i]._fitness,
                _biases = populations[i]._biases,
                _weights = GetWeightArray(populations[i]._weights),
                _weightsData = GetWeightsData(populations[i]._weights)
            };


            dd.Add(i, dataT);
        }

        List<float> GetWeightArray(List<Matrix<float>> weights)
        {
            List<float> weight = new List<float>();

            for (int i = 0; i < weights.Count; i++)
            {
                for (int x = 0; x < weights[i].RowCount; x++)
                {
                    for (int y = 0; y < weights[i].ColumnCount; y++)
                    {
                        weight.Add(weights[i][x, y]);
                    }
                }
            }

            return weight;
        }

        List<int> GetWeightsData(List<Matrix<float>> weights)
        {
            List<int> data = new List<int>();

            data.Add(weights.Count);

            for (int i = 0; i < weights.Count; i++)
            {
                data.Add(weights[i].RowCount);
                data.Add(weights[i].ColumnCount);
            }

            return data;
        }

        string nnn = JsonConvert.SerializeObject(dd);

        File.WriteAllText(Application.dataPath + "/savedNNetData.txt", nnn);
    }



    public static List<DataToSave> LoadData()
    {
        List<DataToSave> dataToSave = new List<DataToSave>();

        if (File.Exists(Application.dataPath + "/savedNNetData.txt"))
        {
            //PATH 
            string saveString = File.ReadAllText(Application.dataPath + "/savedNNetData.txt");

            //NEW DICTIONARY TO DESERIALIZE INTO
            Dictionary<int, DataToSave> DTSDict = new Dictionary<int, DataToSave>();

            DTSDict = JsonConvert.DeserializeObject<Dictionary<int, DataToSave>>(saveString);

            //GETTING ALL DataToSave VALUES OUT OF THE DICTIONARY
            var loadedData = (from kvp in DTSDict select kvp.Value).ToList();


            for (int a = 0; a < loadedData.Count; a++)
            {
                loadedData[a]._matrixWeights = newMatrices();

                //CONVERT NEW DATA TO MATRICES
                List<Matrix<float>> newMatrices()
                {
                    //INDEX TO COUNT ROWS AND COLUMNS
                    int startIndex = 1;

                    //INDEX TO COUNT NUMBERS WHILE ADDING THEM TO MATRIX
                    int startNumber = 0;

                    //NEW LIST WITH THE MATRICES
                    List<Matrix<float>> newMatrices = new List<Matrix<float>>();

                    //CREATE AND ADD MATRICES TO THE LIST, FIRST INDEX IN ORDER (0) MEANS THE NUMBER OF MATRICES TO CREATE
                    for (int i = 0; i < loadedData[a]._weightsData[0]; i++)
                    {
                        //BUILD MATRIX WHERE ROWCOUNT IS SECOND NUMBER AFTER NUMBER OF MATRICES AND COLUMN COUNT IS THE THIRD
                        Matrix<float> newMatrix = Matrix<float>.Build.Dense(loadedData[a]._weightsData[startIndex], loadedData[a]._weightsData[startIndex + 1]);

                        //ADD 2 MORE INTS TO PASS TO THE NEXT ROWCOUNT
                        startIndex += 2;

                        //FILLING MATRIX WITH NUMBERS
                        for (int x = 0; x < newMatrix.RowCount; x++)
                        {
                            for (int y = 0; y < newMatrix.ColumnCount; y++)
                            {
                                newMatrix[x, y] = loadedData[a]._weights[startNumber];

                                startNumber++;
                            }
                        }

                        newMatrices.Add(newMatrix);
                    }

                    return newMatrices;
                }
            }

            dataToSave = loadedData;
        }

        return dataToSave;
    }



    public static void DeleteData()
    {
        if (File.Exists(Application.dataPath + "/savedNNetData.txt"))
        {
            string saveString = Application.dataPath + "/savedNNetData.txt";

            File.Delete(saveString);
        }
    }
}



public class DataToSave
{
    public float _fitness;
    public List<float> _biases;
    public List<float> _weights;
    public List<int> _weightsData;

    public List<Matrix<float>> _matrixWeights;
}