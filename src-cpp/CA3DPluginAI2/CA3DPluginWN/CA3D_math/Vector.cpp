//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Vector.cpp
//    Project   : Fruit Ball Android
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Edit		: 1397\02\08
//    Time      : 15:30:00
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#include "stdafx.h"
#include "Vector.h"


float Vector2Length(VECTOR2 *v)
{
	return sqrtf((v->x * v->x) + (v->y * v->y));
}

float Vector2Dot(VECTOR2 *v1, VECTOR2 *v2)
{
	return (v1->x * v2->x) + (v1->y * v2->y);
}

float Vector2Normalize(VECTOR2 * out, VECTOR2 * v)
{
	float len = Vector2Length(v);

	if (len != 0.0f)
	{
		float m = 1.0f / len;
		*out = (*v) * m;
	}

	return len;
}

// use in Obj file
void Vector3Different(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2)
{
	out->x = v1->x - v2->x;
	out->y = v1->y - v2->y;
	out->z = v1->z - v2->z;
}

float Vector3Length(VECTOR3 *v)
{
	return sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));
}

float Vector3Dot(VECTOR3 *v1, VECTOR3 *v2)
{
	return (v1->x * v2->x) + (v1->y * v2->y) + (v1->z * v2->z);
}

float Vector3Distance(VECTOR3 *v1, VECTOR3 *v2)
{
	VECTOR3 sub = *v1 - *v2;
	return Vector3Length(&sub);
}

float Vector3Normalize(VECTOR3 *out, VECTOR3 *v)
{
	float len = Vector3Length(v);

	if (len)
	{
		float m = 1.0f / len;
		*out = (*v) * m;
	}

	return len;
}

void Vector3Cross(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2)
{
	VECTOR3 v;

	v.x = (v1->y * v2->z) - (v2->y * v1->z);
	v.y = (v1->z * v2->x) - (v2->z * v1->x);
	v.z = (v1->x * v2->y) - (v2->x * v1->y);

	*out = v;
}

void Vector3Midle(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2)
{
	*out = (*v1 + *v2) * 0.5f;
}

void Vector3Invert(VECTOR3 *out, VECTOR3 *v)
{
	*out = -(*v);
}

void Vector3Lerp(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2, float t)
{
	if (t == 1.0f)
	{
		*out = *v2;

		return;
	}
	else if (t == 0.0f)
	{
		*out = *v1;

		return;
	}

	*out = *v1 + (*v2 - *v1) * t;
}

void Vector3Lerp2(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2, float t)
{
	*out = *v1 * t + *v2 * (1.0f - t);
}

void Vector3MultiplyVector3(VECTOR3 *out, VECTOR3 *v1, VECTOR3 *v2)
{
	out->x = v1->x * v2->x;
	out->y = v1->y * v2->y;
	out->z = v1->z * v2->z;
}

void Vector3RotateVector4(VECTOR3 *out, VECTOR3 *v1, VECTOR4 *v2)
{
	VECTOR4 i, t, f;

	Vector4Conjugate(&i, v2);
	Vector4Normalize(&i, &i);
	Vector4MultiplyVector3(&t, v2, v1);
	Vector4MultiplyVector4(&f, &t, &i);

#ifndef ARM
	*out = f;
#else
	out->x = f.x;
	out->y = f.y;
	out->z = f.z;
#endif
}

void Vector3ToRecast(VECTOR3 *v)
{
	float tmp = -v->y;
	v->y = v->z;
	v->z = tmp;
}

void RecastToVector3(VECTOR3 *v)
{
	float tmp = v->y;
	v->y = -v->z;
	v->z = tmp;
}

void Vector4BuildW(VECTOR4 * v)
{
	float l = 1.0f - (v->x * v->x) - (v->y * v->y) - (v->z * v->z);

	v->w = (l < 0.0f) ? 0.0f : -sqrtf(l);
}

float Vector4Dot(VECTOR4 *v1, VECTOR4 *v2)
{
	return ((v1->x * v2->x) + (v1->y * v2->y) + (v1->z * v2->z) + (v1->w * v2->w));
}

float Vector4Length(VECTOR4 *v)
{
	return sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z) + (v->w * v->w));
}

float Vector4Normalize( VECTOR4 *out, VECTOR4 *v)
{
	float len = Vector4Length(v);
	float m = len ? 1.0f / len : 0.0f;
	*out = *v * m;
	
	return len;
}

void Vector4MultiplyVector3(VECTOR4 *out, VECTOR4 *v1, VECTOR3 *v2)
{
	VECTOR4 v;

	v.x = (v1->w * v2->x) + (v1->y * v2->z) - (v1->z * v2->y);
	v.y = (v1->w * v2->y) + (v1->z * v2->x) - (v1->x * v2->z);
	v.z = (v1->w * v2->z) + (v1->x * v2->y) - (v1->y * v2->x);
	v.w = -(v1->x * v2->x) - (v1->y * v2->y) - (v1->z * v2->z);

	*out = v;
}

void Vector4MultiplyVector4(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2)
{
	VECTOR4 v;

	v.x = (v1->x * v2->w) + (v1->w * v2->x) + (v1->y * v2->z) - (v1->z * v2->y);
	v.y = (v1->y * v2->w) + (v1->w * v2->y) + (v1->z * v2->x) - (v1->x * v2->z);
	v.z = (v1->z * v2->w) + (v1->w * v2->z) + (v1->x * v2->y) - (v1->y * v2->x);
	v.w = (v1->w * v2->w) - (v1->x * v2->x) - (v1->y * v2->y) - (v1->z * v2->z);

	*out = v;
}

void Vector4Conjugate(VECTOR4 *out, VECTOR4 *v)
{
	out->x = -v->x;
	out->y = -v->y;
	out->z = -v->z;
	out->w = v->w;
}

void Vector4Invert(VECTOR4 *out, VECTOR4 *v)
{
	*out = -(*v);
}

void Vector4Lerp(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2, float t)
{
	float k1, k2;
	float dot = Vector4Dot(v1, v2);
	VECTOR4 tmp = *v2;

	if (t == 1.0f)
	{
		*out = *v2;
		
		return;
	}
	else if (t == 0.0f)
	{
		*out = *v1;
		
		return;
	}

	if (dot < 0.0f)
	{
		tmp = -tmp;
		dot = -dot;
	}


	if (dot > 0.999999f)
	{
		k1 = 1.0f - t;
		k2 = t;
	}
	else
	{
		float s = sqrtf(1.0f - (dot * dot));
		float o1 = atan2f(s, dot);
		float o2 = 1.0f / s;

		k1 = sinf((1.0f - t) * o1) * o2;
		k2 = sinf(t * o1) * o2;
	}

	*out = ((*v1) * k1) + (tmp * k2);
}

void Vector4SLerp(VECTOR4 *out, VECTOR4 *v1, VECTOR4 *v2, float t)
{
	float k1, k2;
	float c = Vector4Dot(v1, v2);
	VECTOR4 tmp = *v2;

	if (t == 1.0f)
	{
		*out = *v2;

		return;
	}
	else if (t == 0.0f)
	{
		*out = *v1;

		return;
	}

	if (c < 0.0f)
	{
		tmp = -tmp;
		c = -c;
	}

	if (c > 0.999999f)
	{
		k1 = 1.0f - t;
		k2 = t;
	}
	else
	{
		float s = sqrtf(1.0f - (c * c));
		float o1 = atan2f(s, c);
		float o2 = 1.0f / s;

		k1 = sinf((1.0f - t) * o1) * o2;
		k2 = sinf(t * o1) * o2;
	}

	*out = ((*v1) * k1) + (tmp * k2);
}

float Vector2::Length(VECTOR2 &v)
{
	return sqrtf((v.x * v.x) + (v.y * v.y));
}

float Vector2::Dot(VECTOR2 &v1, VECTOR2 &v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y);
}

float Vector2::Normalize(VECTOR2 & out, VECTOR2 & v)
{
	float len = sqrtf((v.x * v.x) + (v.y * v.y));

	if (len)
		out = v / len;

	return len;
}

float Vector3::Length(VECTOR3 & v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

float Vector3::Norm2(VECTOR3 & v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

float Vector3::Dot(VECTOR3 &v1, VECTOR3 &v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

VECTOR3 Vector3::Cross(VECTOR3 &v1, VECTOR3 &v2)
{
	VECTOR3 v;

	v.x = (v1.y * v2.z) - (v2.y * v1.z);
	v.y = (v1.z * v2.x) - (v2.z * v1.x);
	v.z = (v1.x * v2.y) - (v2.x * v1.y);

	return v;
}

float Vector3::Normalize(VECTOR3 & out, VECTOR3 & v)
{
	float len = sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z));

	if (len)
		out = v / len;

	return len;
}
