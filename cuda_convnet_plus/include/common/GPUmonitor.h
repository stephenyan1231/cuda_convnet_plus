/*
 * GPUmonitor.h
 *
 *  Created on: Jul 2, 2013
 *      Author: Zhicheng Yan
 */
#ifndef _GPU_MONITOR_H_
#define _GPU_MONITOR_H_

#include <cstdlib>
#include <cstdio>


#define GPU_MEMORY_BUDGET ((long long)6.0e+9)

class GPUmonitor{
	int _verbose;
	long long _usedMemory;
public:
	GPUmonitor();

	void setVerbose(int verbose);

	void addUsedMemory(long long memorySize);

	void freeUsedMemory(long long memorySize);

	long long getUsedMemory();

	void reportUsedMemory();

	void showMemory(long long memoryUse);
};

#endif
