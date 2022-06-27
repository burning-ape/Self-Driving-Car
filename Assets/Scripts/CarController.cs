using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NeuralNetwork))]
public class CarController : MonoBehaviour
{
    private Vector3 _startPosition, _startRotation;
    private NeuralNetwork _network;

    [Range(-1f, 1f)]
    [SerializeField] private float _a, _t;
    [SerializeField] private float _timeSinceStart = 0f;


    [Header("Car Settings")]
    [SerializeField] private float _overallEfficiency;
    [SerializeField] private float _distanceMultipler = 1.4f;
    [SerializeField] private float _avgSpeedMultiplier = 0.2f;
    [SerializeField] private float _sensorMultiplier = 0.1f;


    [Header("Network Settings")]
    public int _LAYERS = 1;
    public int _NEURONS = 10;
    [SerializeField] private GeneticManager _geneticManager;


    private Vector3 _lastPosition;
    private float _totalDistanceTravelled;
    private float _avgSpeed;

    private float _aSensor, _bSensor, _cSensor;
    private Vector3 _inp;

    private void Awake()
    {
        //HASH START POSITION AND ROTATION
        _startPosition = transform.position;
        _startRotation = transform.eulerAngles;

        _network = GetComponent<NeuralNetwork>();
    }



    private void FixedUpdate()
    {
        //SENSORS DETECT THE DISTANCE TO THE WALLS
        InputSensors();

        _lastPosition = transform.position;

        //RUNNING NETWORK
        (_a, _t) = _network.RunNetwork(_aSensor, _bSensor, _cSensor);


        //MOVING CAR BY OUTPUT RESULTS OF NETWORK
        MoveCar(_a, _t);

        //COUNT TIME SINCE START
        _timeSinceStart += Time.deltaTime;

        //CALCULATING
        CalculateEfficiency();
    }



    private void OnCollisionEnter(Collision collision)
    {
        //IF CAR TOUCHES THE WALL, CALL DEATH
        Death();
    }



    private void Death()
    {
        _geneticManager.Death(_overallEfficiency, _network);
    }



    public void ResetWithNetwork(NeuralNetwork net)
    {
        _network = net;
        Reset();
    }



    public void Reset()
    {
        _timeSinceStart = 0f;
        _totalDistanceTravelled = 0f;
        _avgSpeed = 0f;
        _lastPosition = _startPosition;
        _overallEfficiency = 0f;
        transform.position = _startPosition;
        transform.eulerAngles = _startRotation;
    }



    private void CalculateEfficiency()
    {
        //ADD DISTANCE FROM THE POSITION OF THE CAR FROM THE PREVIOUS MOVE TO THE CURRENT
        _totalDistanceTravelled += Vector3.Distance(transform.position, _lastPosition);

        _avgSpeed = _totalDistanceTravelled / _timeSinceStart;

        _overallEfficiency = (_totalDistanceTravelled * _distanceMultipler) + (_avgSpeed * _avgSpeedMultiplier) + (((_aSensor + _bSensor + _cSensor) / 3) * _sensorMultiplier);

        //IF EFFICIENCY OF THE NEURAL NETWORK IS TOO LITTLE FOR THE GIVEN TIME, CALL DEATH
        if (_timeSinceStart > 20 && _overallEfficiency < 40)
        {
            Death();
        }

        //IF EFFICIENCY OF THE NEURAL NETWORK IS GOOD, CALL DEATH
        if (_overallEfficiency >= 2000)
        {

            Death();
        }
    }



    private void InputSensors()
    {
        Vector3 a = (transform.forward + transform.right);
        Vector3 b = (transform.forward);
        Vector3 c = (transform.forward - transform.right);


        Ray r = new Ray(transform.position, a);
        RaycastHit hit;

        if (Physics.Raycast(r, out hit))
        {
            _aSensor = hit.distance / 20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        r.direction = b;

        if (Physics.Raycast(r, out hit))
        {
            _bSensor = hit.distance / 20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        r.direction = c;

        if (Physics.Raycast(r, out hit))
        {
            _cSensor = hit.distance / 20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

    }



    public void MoveCar(float v, float h)
    {
        _inp = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, v * 11.4f), 0.02f);
        _inp = transform.TransformDirection(_inp);

        transform.position += _inp;

        transform.eulerAngles += new Vector3(0, (h * 90) * 0.02f, 0);
    }
}