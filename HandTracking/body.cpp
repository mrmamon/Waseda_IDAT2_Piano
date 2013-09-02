#pragma once
#include "header.h"
#include <vector>

computingClass::computingClass(int* pInt,int arrSize)
{
	for(int i=0;i<arrSize;i++)
	{
		vec.push_back(pInt[i]);
	}
}