//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Ray.cpp
//    Project   : Fruit Ball
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Time      : --:--:--
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------
#include "stdafx.h"
#include "Ray.h"

CRay::CRay()
{
	org = VECTOR3();
	dir = VECTOR3();
}

CRay::CRay(VECTOR3 o, VECTOR3 d)
{
	org = o;
	dir = d;
}

void CRay::UpdateNormal()
{
	nor = dir - org;
	Vector3Normalize(&nor, &nor);
}
