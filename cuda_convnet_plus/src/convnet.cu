/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * updated by zyan3
 * */
#include <vector>
#include <iostream> 
#include <string>

#include <nvmatrix.cuh>
#include <nvmatrix_operators.cuh>
#include <matrix.h>
#include <convnet.cuh>
#include <util.cuh>

using namespace std;

extern bool verbose;

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp) {
#ifdef _WIN32
	return (bool)( (pProp->tccDriver && (pProp->major >= 2)) ? true : false);
#else
	return (bool) (pProp->major >= 2);
#endif
}

/* 
 * =======================
 * ConvNet
 * =======================
 */

int ConvNet::getA() {
	return 1;
}

ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID,
		int numDevices) :
		Thread(false), _deviceID(deviceID), _numDevices(numDevices), _data(NULL) {
	try {
		int numLayers = PyList_GET_SIZE(layerParams);

		for (int i = 0; i < numLayers; i++) {

			PyObject* paramsDict = PyList_GET_ITEM(layerParams, i);
			string layerType = pyDictGetString(paramsDict, "type");
			printf("ConvNet::ConvNet init layers: %d %s\n", i,
					layerType.c_str());

			Layer* l = initLayer(layerType, paramsDict);
			// Connect backward links in graph for this layer
			intv* inputLayers = pyDictGetIntV(paramsDict, "inputs");
			if (inputLayers != NULL) {
				for (int i = 0; i < inputLayers->size(); i++) {
					l->addPrev(&getLayer(inputLayers->at(i)));
				}
			}

			delete inputLayers;
		}

		// Connect the forward links in the graph
		for (int i = 0; i < _layers.size(); i++) {
			vector<Layer*>& prev = _layers[i]->getPrev();
			for (int j = 0; j < prev.size(); j++) {
				prev[j]->addNext(_layers[i]);
			}
		}

#ifdef MULTIGPU
		for (int i = 0; i < _layers.size(); i++) {
			_layers[i]->initBpropEvent();
		}
#endif

		// Execute post-initialization stuff
		for (int i = 0; i < _layers.size(); i++) {
			_layers[i]->postInit();
		}

		_dp = new DataProvider(minibatchSize);
	} catch (string& s) {
		cout << "Error creating ConvNet: " << s << endl;
		exit(1);
	}
}

ConvNet::~ConvNet() {
	cublasDestroy(_cudaHandle);
}

/*
 * Override this in derived classes
 */
Layer* ConvNet::initLayer(string& layerType, PyObject* paramsDict) {
	if (layerType == "fc") {
		_layers.push_back(new FCLayer(this, paramsDict));
	} else if (layerType == "conv") {
		_layers.push_back(new ConvLayer(this, paramsDict));
	} else if (layerType == "local") {
		_layers.push_back(new LocalUnsharedLayer(this, paramsDict));
	} else if (layerType == "pool") {
		_layers.push_back(&PoolLayer::makePoolLayer(this, paramsDict));
	} else if (layerType == "rnorm") {
		_layers.push_back(new ResponseNormLayer(this, paramsDict));
	} else if (layerType == "cmrnorm") {
		_layers.push_back(new CrossMapResponseNormLayer(this, paramsDict));
	} else if (layerType == "cnorm") {
		_layers.push_back(new ContrastNormLayer(this, paramsDict));
	} else if (layerType == "softmax") {
		_layers.push_back(new SoftmaxLayer(this, paramsDict));
	} else if (layerType == "logsoftmax") {
		_layers.push_back(new LogSoftmaxLayer(this, paramsDict));
	} else if (layerType == "eltsum") {
		_layers.push_back(new EltwiseSumLayer(this, paramsDict));
	} else if (layerType == "eltmax") {
		_layers.push_back(new EltwiseMaxLayer(this, paramsDict));
	} else if (layerType == "neuron") {
		_layers.push_back(new NeuronLayer(this, paramsDict));
	} else if (layerType == "nailbed") {
		_layers.push_back(new NailbedLayer(this, paramsDict));
	} else if (layerType == "blur") {
		_layers.push_back(new GaussianBlurLayer(this, paramsDict));
	} else if (layerType == "resize") {
		_layers.push_back(new ResizeLayer(this, paramsDict));
	} else if (layerType == "rgb2yuv") {
		_layers.push_back(new RGBToYUVLayer(this, paramsDict));
	} else if (layerType == "rgb2lab") {
		_layers.push_back(new RGBToLABLayer(this, paramsDict));
	} else if (layerType == "data") {
		DataLayer *d = new DataLayer(this, paramsDict);
		_layers.push_back(d);
		_dataLayers.push_back(d);
	} else if (layerType == "crop") {
		_layers.push_back(new CroppingLayer(this, paramsDict));
	} else if (layerType == "scaling") {
		_layers.push_back(new ScalingLayer(this, paramsDict));
	} else if (layerType == "normalize") {
		_layers.push_back(new NormalizeLayer(this, paramsDict));
	} else if (layerType == "concatenate"){
		_layers.push_back(new ConcatenateLayer(this, paramsDict));
	}
	else if (strncmp(layerType.c_str(), "cost.", 5) == 0) {
		CostLayer *c = &CostLayer::makeCostLayer(this, layerType, paramsDict);
		_layers.push_back(c);
		_costs.push_back(c);
	} else {
		throw string("Unknown layer type ") + layerType;
	}

	return _layers.back();
}
/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNet::initCuda() {
	printf("run on GPU : %d\n",
			_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
	cudaSetDevice(_deviceID < 0 ? cutGetMaxGflopsDeviceId() : _deviceID);
	cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);

	cublasStatus_t stat = cublasCreate(&_cudaHandle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		exit(1);
	}

#ifdef MULTIGPU
	// check Peer-2-Peer capability
	if(_numDevices>1) {
		// Query device properties
		cudaDeviceProp prop[64];

		// assume devices ID are (0,1,..,_numDevices-1)
		for (int i=0; i < _numDevices; i++)
		{
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
			// Only boards based on Fermi can support P2P
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
		}

		for(int i=0;i<_numDevices;++i) {
			for(int j=0;j<_numDevices;++j) {
				if(j!=i) {
					int access_p0_p1, access_p1_p0;
					checkCudaErrors(cudaDeviceCanAccessPeer(&access_p0_p1, i, j));
					checkCudaErrors(cudaDeviceCanAccessPeer(&access_p1_p0, j, i));
					// Output results from P2P capabilities
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name, i,
							prop[j].name, j , access_p0_p1 ? "Yes" : "No");
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[j].name, j,
							prop[i].name, i , access_p1_p0 ? "Yes" : "No");
					if(access_p0_p1) {
						checkCudaErrors(cudaSetDevice(i));
						checkCudaErrors(cudaDeviceEnablePeerAccess(j,0));
					}
					if(access_p1_p0) {
						checkCudaErrors(cudaSetDevice(j));
						checkCudaErrors(cudaDeviceEnablePeerAccess(i,0));
					}
				}
			}
		}
		// check UVA capability
		bool UVA = 1;
		for(int i=0;i<_numDevices;++i) {
			printf("GPU%d supports UVA: %s\n",i,prop[i].unifiedAddressing ? "Yes":"No");
			UVA=UVA && (prop[i].unifiedAddressing == 1);
		}
		if(!UVA) {
			printf("UVA is not supported. exit!\n");
			exit(0);
		}
	}

	for(int i=0;i<_numDevices;++i) {
		checkCudaErrors(cudaSetDevice(i));
		NVMatrix::initRandom(time(0));
	}
	checkCudaErrors(cudaSetDevice(DEFAULT_GPU));
#endif

	copyToGPU();
}

void* ConvNet::run() {
	initCuda();

	while (true) {
		Worker* worker = _workerQueue.dequeue();
		worker->run();
		delete worker;
	}
	return NULL;
}

Queue<Worker*>& ConvNet::getWorkerQueue() {
	return _workerQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
	return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
	return *_dp;
}

Layer& ConvNet::operator[](int idx) {
	return *_layers[idx];
}

Layer& ConvNet::getLayer(int idx) {
	return *_layers[idx];
}

void ConvNet::copyToCPU() {
	for (int i = 0; i < _layers.size(); i++) {
		_layers[i]->copyToCPU();
	}
}

void ConvNet::copyToGPU() {
	for (int i = 0; i < _layers.size(); i++) {
		_layers[i]->copyToGPU();
	}
}

void ConvNet::updateWeights() {
	for (int i = 0; i < _layers.size(); i++) {
		_layers[i]->updateWeights();
	}
}

void ConvNet::reset() {
	for (int i = 0; i < _layers.size(); i++) {
		_layers[i]->reset();
	}
}

int ConvNet::getNumLayers() {
	return _layers.size();
}

void ConvNet::bprop(PASS_TYPE passType) {
	for (int i = 0; i < _costs.size(); i++) {
		_costs[i]->bprop(passType);
	}
	reset();
}

void ConvNet::fprop(PASS_TYPE passType) {
	assert(_data != NULL);
	reset();
	for (int i = 0; i < _dataLayers.size(); i++) {
		_dataLayers[i]->fprop(_data->getData(), passType);
	}
}

void ConvNet::fprop(GPUData& data, PASS_TYPE passType) {
	if (&data != _data) {
		delete _data;
	}
	_data = &data;
	fprop(passType);
}

void ConvNet::fprop(int miniIdx, PASS_TYPE passType) {
	delete _data;
	_data = &_dp->getMinibatch(miniIdx);
	fprop(passType);
}

Cost& ConvNet::getCost() {
	return *new Cost(_data->getNumCases(), _costs);
}

// Same as getCost() but adds results to given cost and returns it
Cost& ConvNet::getCost(Cost& cost) {
	Cost& newCost = getCost();
	cost += newCost;
	delete &newCost;
	return cost;
}

double ConvNet::getCostValue() {
	Cost& cost = getCost();
	double val = cost.getValue();
	delete &cost;
	return val;
}

/*
 * Gradient checking stuff
 */
void ConvNet::checkGradients() {
	_numFailures = 0;
	_numTests = 0;
	fprop(0, PASS_GC);
	_baseErr = getCostValue();
	bprop(PASS_GC);

	for (vector<Layer*>::iterator it = _layers.begin(); it != _layers.end();
			++it) {
		(*it)->checkGradients();
	}

	cout << "------------------------" << endl;
	if (_numFailures > 0) {
		cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
	} else {
		cout << "ALL " << _numTests << " TESTS PASSED" << endl;
	}
}

/*
 * name: weight matrix name
 * eps: finite difference step
 */
bool ConvNet::checkGradient(const string& name, float eps, Weights& weights) {
	Matrix numGrad(weights.getNumRows(), weights.getNumCols());
	Matrix diff(numGrad);
	numGrad.apply(Matrix::ZERO);
	Matrix weightsCPU;
	if(verbose)
		printf("weights.getW().copyToHost\n");
	weights.getW().copyToHost(weightsCPU, true);

	for (int i = 0; i < weights.getNumRows(); i++) {
		for (int j = 0; j < weights.getNumCols(); j++) {
			float v = weightsCPU(i, j);
			weightsCPU(i, j) += eps;
			weights.getW().copyFromHost(weightsCPU);
			weightsCPU(i, j) = v;
			fprop(PASS_GC);
			double err = getCostValue();
			numGrad(i, j) = (err - _baseErr) / (_data->getNumCases() * eps);
			if (isnan(numGrad(i,j)) || isinf(numGrad(i,j))) {
				cout
						<< "Numerical computation produced nan or inf when checking '"
						<< name << "': " << numGrad(i, j) << endl;
				cout
						<< "Consider reducing the sizes of the weights or finite difference steps."
						<< endl;
				cout << "Exiting." << endl;
				exit(1);
			}
			weights.getW().copyFromHost(weightsCPU);
		}
	}

	Matrix gradCPU;
	if(verbose)
		printf("weights.getGrad().copyToHost\n");
	weights.getGrad().copyToHost(gradCPU, true);
	gradCPU.scale(-1.0 / _data->getNumCases());
	float analNorm = gradCPU.norm();
	float numNorm = numGrad.norm();
	numGrad.subtract(gradCPU, diff);
	float relErr = diff.norm() / analNorm;
	bool fail = relErr >= GC_REL_ERR_THRESH;
	if (fail || !GC_SUPPRESS_PASSES) {
		cout << "========================" << endl;
		printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS",
				name.c_str());
		cout << "========================" << endl;
		cout << "Analytic:" << endl;
		gradCPU.print(6, 4);
		cout << "Numeric:" << endl;
		numGrad.print(6, 4);
		printf("Analytic norm: %e\n", analNorm);
		printf("Numeric norm:  %e\n", numNorm);
		printf("Relative error: %e\n", relErr);
	}
	_numTests++;
	_numFailures += fail;
	return fail;
}

cublasHandle_t ConvNet::getCublasHandle() {
	return _cudaHandle;
}

void ConvNet::scaleCostLayerCoeff(float scale) {
	for (int i = 0; i < _costs.size(); ++i) {
		_costs[i]->scaleCoeff(scale);
	}
}

void ConvNet::setWeightsLayerEpsScale(float scale) {
//	printf("ConvNet::setWeightsLayerEpsScale scale:%f\n",scale);
	for (int i = 0; i < _layers.size(); ++i) {
		_layers[i]->setWeightsEpsScale(scale);
	}
}

void ConvNet::multiplyWeightsLayerEps(float multiplier) {
	printf("\n-------ConvNet::multiplyWeightsLayerEps multiplier:%f---------\n",
			multiplier);
	for (int i = 0; i < _layers.size(); ++i) {
		_layers[i]->multiplyWeightsEpsScale(multiplier);
	}
}

