using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;


public class GeneticManager : MonoBehaviour
{
    [SerializeField] private CarController _controller;

    [Header("Controls")]
    [SerializeField] private int _totalPopulation;

    [Range(0.0f, 1.0f)]
    [SerializeField] private float _mutationRate = 0.055f;

    [Header("Crossover Controls")]
    [SerializeField] private int _bestAgentSelection = 6;
    [SerializeField] private int _numberToCrossover;

    private List<int> _genePool = new List<int>();

    private List<NeuralNetwork> _population = new List<NeuralNetwork>();

    [Header("Public View")]
    [SerializeField] private int _currentGeneration;
    [SerializeField] private int _currentGenome = 1;



    [Range(1f, 100f)]
    [SerializeField] private float _timeScale = 1f;

    [SerializeField] private bool _LoadPreviousResults = false;
    [SerializeField] private bool _startWithDeletingPreviousData = false;



    private void Start()
    {
        if (_LoadPreviousResults) { LoadPopulation(); }
        else if (_startWithDeletingPreviousData) { CreatePopulation(); DeletePreviousData(); }
        else { CreatePopulation(); }
    }



    private void FixedUpdate() => Time.timeScale = _timeScale;



    private void DeletePreviousData()
    {
        SaveAndLoadSystem.DeleteData();
    }


    private void CreatePopulation()
    {
        FillPopulationWithRandomValues(_population, 0);
        ResetToCurrentGenome();
    }



    private void FillPopulationWithRandomValues(List<NeuralNetwork> newPopulation, int _startingIndex)
    {
        while (_startingIndex < _totalPopulation)
        {
            newPopulation.Add(new NeuralNetwork());
            newPopulation[_startingIndex].Initialise(_controller._LAYERS, _controller._NEURONS);
            _startingIndex++;
        }

    }



    private void LoadPopulation()
    {
        List<DataToSave> dataToSave = SaveAndLoadSystem.LoadData();
        SetPopulationWithLoadedData(_population, 0, dataToSave);
    }



    private void SetPopulationWithLoadedData(List<NeuralNetwork> newPopulation, int _startingIndex, List<DataToSave> loadedData)
    {
        while (_startingIndex < _totalPopulation)
        {
            newPopulation.Add(new NeuralNetwork());
            newPopulation[_startingIndex].Initialise(_controller._LAYERS, _controller._NEURONS);

            newPopulation[_startingIndex]._weights = loadedData[_startingIndex]._matrixWeights;
            newPopulation[_startingIndex]._biases = loadedData[_startingIndex]._biases;

            _startingIndex++;
        }

        ResetToCurrentGenome();
    }



    private void ResetToCurrentGenome()
    {
        _controller.ResetWithNetwork(_population[_currentGenome]);
    }



    private void RePopulate()
    {
        _genePool.Clear();
        _currentGeneration++;

        List<NeuralNetwork> newPopulation = PickBestPopulation();
        Mutate(newPopulation);
        Crossover(newPopulation);

        for(int i = 0; i < newPopulation.Count; i++)
        {
            Debug.Log("  BIAS:  " + newPopulation[i]._biases[0] + "  WEIGHTS:  " + newPopulation[i]._weights[0][0, 0]);
        }

        _population.Clear();
        _population = newPopulation;

        _currentGenome = 0;

        ResetToCurrentGenome();
    }



    private List<NeuralNetwork> PickBestPopulation()
    {
        List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();

        Dictionary<float, NeuralNetwork> effDict = new Dictionary<float, NeuralNetwork>();

        //ADD POPULATION AND ITS EFFICIENCY TO THE DICTIONARY
        for (int i = 0; i < _totalPopulation; i++)
        {
            //PREVENT ADDING THE SAME KEY TO THE DICTIONARY
            if (effDict.ContainsKey(_population[i]._fitness)) _population[i]._fitness += Random.Range(-2.000f, 2.000f);

            effDict.Add(_population[i]._fitness, _population[i]);
        }

        //SORT DICTIONARY BY KEYS
        var sortedDict = from entry in effDict orderby entry.Key ascending select entry;

        //REVERSE DICTIONARY
        var reversedDict = sortedDict.Reverse().ToArray();

        //GET LIST OF VALUES (NEURAL NETWORK)
        var mostEfficient = (from kvp in reversedDict select kvp.Value).ToList();

        //ADD MOST EFFICIENT POPULATIONS TO THE LIST
        for (int i = 0; i < _bestAgentSelection; i++)
        {
            mostEfficient[i]._fitness = 0f;
            newPopulation.Add(mostEfficient[i]);
        }

        effDict.Clear();

        return newPopulation;
    }



    private void Crossover(List<NeuralNetwork> newPopulation)
    {
        List<NeuralNetwork> crossedNetworks = new List<NeuralNetwork>();

        for (int i = 0; i < _numberToCrossover; i++)
        {
            //CREATE NEW CHILD
            NeuralNetwork child = new NeuralNetwork();
            child.Initialise(_controller._LAYERS, _controller._NEURONS);
            child._fitness = 0;

            //CROSS WEIGHTS AND BIASES INTO CHILD OBJECT
            child._weights = newPopulation[i + 1]._weights;
            child._biases = newPopulation[i]._biases;

            crossedNetworks.Add(child);
        }

        newPopulation.AddRange(crossedNetworks);
    }



    private void Mutate(List<NeuralNetwork> newPopulation)
    {
        for (int i = 0; i < newPopulation.Count; i++)
        {
            for (int y = 0; y < newPopulation[i]._weights.Count; y++)
            {
                if (Random.Range(0.0f, 1.0f) < _mutationRate)
                {
                    newPopulation[i]._weights[y] = MutateMatrix(newPopulation[i]._weights[y]);
                }
            }
        }
    }



    Matrix<float> MutateMatrix(Matrix<float> A)
    {
        Matrix<float> C = A;

        for (int i = 0; i < 3; i++)
        {
            int randomColumn = Random.Range(0, C.ColumnCount);
            int randomRow = Random.Range(0, C.RowCount);

            C[randomRow, randomColumn] = Mathf.Clamp(C[randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return C;
    }



    public void Death(float fitness, NeuralNetwork network)
    {
        if (_currentGenome < _population.Count - 1)
        {
            _population[_currentGenome]._fitness = fitness;
            _currentGenome++;
            ResetToCurrentGenome();
        }
        else
        {
            _population[_currentGenome]._fitness = fitness;
            SaveAndLoadSystem.SaveData(_population);
            RePopulate();
        }

    }

}
