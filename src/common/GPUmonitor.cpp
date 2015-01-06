/*
 * GPUmonitor.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zhicheng Yan
 */

#include <GPUmonitor.h>

GPUmonitor::GPUmonitor(){
	_verbose=0;
	_usedMemory=0;
}


void GPUmonitor::setVerbose(int verbose){
	_verbose = verbose;
}

void GPUmonitor::addUsedMemory(long long memorySize){
	_usedMemory += memorySize;
	if (_verbose>0) {
		printf("GPUmonitor: addUsedMemory: ");
		showMemory(memorySize);
		reportUsedMemory();
	}
}

void GPUmonitor::freeUsedMemory(long long memorySize){
	if(_usedMemory<memorySize){
		fprintf(stderr,"_usedMemory<memorySize _usedMemory:%lld memorySize:%lld",
				_usedMemory,memorySize);
		exit(1);
	}
	_usedMemory-=memorySize;
	if (_verbose) {
		printf("GPUmonitor: freeUsedMemory: ");
		showMemory(memorySize);
		reportUsedMemory();
	}
}

long long GPUmonitor::getUsedMemory(){
	return _usedMemory;
}

void GPUmonitor::reportUsedMemory(){
	printf("GPU memory usage: ");
	showMemory(_usedMemory);
}

void GPUmonitor::showMemory(long long memoryUse){
	if(memoryUse<1e+3)
		printf("%lld B\n",memoryUse);
	else if(memoryUse<1e+6)
		printf("%lld KB\n",memoryUse/(long long)(1e+3L));
	else
		printf("%lld MB\n",memoryUse/(long long)(1e+6L));
}
