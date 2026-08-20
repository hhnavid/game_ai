//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Ray.hpp
//    Project   : Fruit Ball
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Time      : --:--:--
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#pragma once

#include "Types.h"
#include "Vector.h"

class CRay
{
public:
	CRay();
	CRay(VECTOR3 o, VECTOR3 d);

	void UpdateNormal();

public:
	VECTOR3 org; // origin
	VECTOR3 dir; // direction
	VECTOR3 nor; // normal
};