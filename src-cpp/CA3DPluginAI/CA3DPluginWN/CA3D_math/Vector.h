//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Vector.h
//    Project   : Fruit Ball Android
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Edit		: 1397\02\08
//    Time      : 14:44:00
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#pragma once

#include "Types.h"

// vector2
float Vector2Length(VECTOR2 *v);
float Vector2Dot(VECTOR2 *v1, VECTOR2 *v2);
float Vector2Normalize(VECTOR2 *out, VECTOR2 *v);

// vector3
void Vector3ToRecast(VECTOR3 *v);
void RecastToVector3(VECTOR3 *v);
void Vector3Invert(VECTOR3 *out, VECTOR3 *v);
void Vector3Cross(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2);
void Vector3Midle(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2);
void Vector3Different(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2);
void Vector3RotateVector4(VECTOR3 *out, VECTOR3 *v1, VECTOR4 *v2);
void Vector3Lerp(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2, float t);
void Vector3Lerp2(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2, float t);
void Vector3MultiplyVector3(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2);

float Vector3Length(VECTOR3 *v);
float Vector3Dot(VECTOR3 *v1, VECTOR3 *v2);
float Vector3Distance(VECTOR3 *v1, VECTOR3 *v2);
float Vector3Normalize(VECTOR3 *out, VECTOR3 *v);

// vector4
void Vector4BuildW(VECTOR4 *v);
void Vector4Invert(VECTOR4 *out, VECTOR4 *v);
void Vector4Conjugate(VECTOR4 *out, VECTOR4 *v);
void Vector4Lerp(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2, float t);
void Vector4SLerp(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2, float t);
void Vector4MultiplyVector3(VECTOR4 *out, VECTOR4 *v1, VECTOR3 *v2);
void Vector4MultiplyVector4(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2);

float Vector4Length(VECTOR4 *v);
float Vector4Dot(VECTOR4 *v1, VECTOR4 *v2);
float Vector4Normalize(VECTOR4 *out, VECTOR4 *v);

class Vector2
{
public:
	static float Length(VECTOR2 &v);
	static float Dot(VECTOR2 &v1, VECTOR2 &v2);
	static float Normalize(VECTOR2 & out, VECTOR2 & v);
};

class Vector3
{
public:
	static float Norm2(VECTOR3 &v);
	static float Length(VECTOR3 &v);
	static float Dot(VECTOR3 &v1, VECTOR3 &v2);
	static VECTOR3 Cross(VECTOR3 &v1, VECTOR3 &v2);
	static float Normalize(VECTOR3 & out, VECTOR3 & v);
};